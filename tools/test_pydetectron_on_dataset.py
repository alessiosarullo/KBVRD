import argparse
import os
import os.path as osp
import sys
from collections import defaultdict
import numpy as np

import torch
import torchvision
import cv2

from torchvision.transforms import Lambda, ToPILImage
from lib.drivers.datasets import SquarePad, Resize, ToTensor, Normalize, Compose, im_list_to_4d_tensor
from lib.pydetectron_api.detection import im_detect_all_with_feats

sys.path.insert(0, osp.abspath(osp.join('pydetectron', 'lib')))
# Note: fixing these imports doesn't work if Detectron's ones stay the same, because they end up looking at different configs (for some reason).
import nn as mynn
from core.config import cfg, cfg_from_file, assert_and_infer_cfg
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as dummy_datasets
import utils.misc as misc_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer
from core.test import segm_results
import utils.vis as vis_utils
from utils.blob import prep_im_for_blob
sys.path.remove(osp.abspath(osp.join('pydetectron', 'lib')))


def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')

    parser.add_argument('--cfg', dest='cfg_file', help='config file', required=True)
    parser.add_argument('--load_detectron', help='path to the detectron weight pickle file', required=True)
    parser.add_argument('--image_dir', help='directory to load images for demo', required=True)
    parser.add_argument('--output_dir', help='directory to save demo results', default="detectron_outputs")

    return parser.parse_args()


def main():
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")
    device = torch.device('cuda')

    # model = 'e2e_mask_rcnn_R-50-FPN_1x'
    model = 'e2e_mask_rcnn_R-50-C4_2x'
    sys.argv += ['--cfg', 'pydetectron/configs/baselines/%s.yaml' % model,
                 '--load_detectron', 'data/pretrained_model/%s.pkl' % model,
                 '--image_dir', '/media/alex/Woodo/PhD/KB-HOI data/fake_HICO-DET',
                 '--output_dir', 'detectron_outputs/test_ds/',
                 ]

    args = parse_args()
    print('Called with args:')
    print(args)

    dataset = dummy_datasets.get_coco_dataset()
    cfg.MODEL.NUM_CLASSES = len(dataset.classes)

    print('load cfg from file: {}'.format(args.cfg_file))
    cfg_from_file(args.cfg_file)
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()
    maskRCNN.cuda()
    maskRCNN.eval()

    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    imglist = misc_utils.get_imagelist_from_dir(args.image_dir)
    num_images = len(imglist)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    batch_size = 2

    num_images = (num_images // batch_size) * batch_size
    images = []
    batches, batch, im_index = [], [], []
    for i in range(num_images):
        print('Img', i)
        im = cv2.imread(imglist[i])
        images.append(im)
        im_index.append(i)
        ims, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, [cfg.TEST.SCALE], cfg.TEST.MAX_SIZE)
        assert len(ims) == 1
        im = np.transpose(ims[0], axes=(2, 0, 1))  # to BGR
        im_scale = np.squeeze(im_scale)
        print(im_scale)

        # im = Image.open(imglist[i]).convert('RGB')
        # img_w, img_h = im.size
        # img_scale_factor = cfg.TEST.SCALE / min(img_h, img_w)
        # im_info = np.array([*im.shape[1:], img_scale_factor])

        batch.append({'img': torch.tensor(im, device=device),
                      'im_scale': im_scale})
        if (i + 1) % batch_size == 0:
            im_batch = im_list_to_4d_tensor([b['img'] for b in batch]).to(device=device)
            im_scales = np.array([b['im_scale'] for b in batch])
            im_infos = np.concatenate([np.tile(im_batch.shape[2:], reps=[im_scales.size, 1]), im_scales[:, None]], axis=1)
            print('>>', im_infos, im_batch.shape)
            batches.append({'image_idxs': im_index,
                            'input': {'data': im_batch,
                                      'im_info': torch.Tensor(im_infos),}
                            })
            batch = []
            im_index = []

    timers = defaultdict(Timer)
    for batch_i, batch in enumerate(batches):
        print('Batch', batch_i)

        inputs = batch['input']
        scores, boxes, box_classes, im_ids, masks, feat_map = im_detect_all_with_feats(maskRCNN, inputs, timers=timers)

        timers['device_transfer'].tic()
        masks = masks.cpu().numpy()
        timers['device_transfer'].toc()

        boxes_with_scores = np.concatenate([boxes, scores[:, None]], axis=1)
        for i in np.unique(im_ids):
            im_idx = batch['image_idxs'][i]
            im = images[im_idx]
            mask_i = (im_ids == i)
            boxes_i = boxes_with_scores[mask_i, :]
            box_classes_i = box_classes[mask_i]

            cls_boxes = [[]] + [boxes_i[box_classes_i == j, :] for j in range(1, cfg.MODEL.NUM_CLASSES)]
            timers['misc_mask'].tic()
            cls_segms = segm_results(cls_boxes, masks, boxes, im.shape[0], im.shape[1])
            timers['misc_mask'].toc()

            im_name, _ = os.path.splitext(os.path.basename(imglist[im_idx]))
            vis_utils.vis_one_image(
                im[:, :, ::-1],  # BGR -> RGB for visualization
                im_name,
                args.output_dir,
                cls_boxes,
                cls_segms,
                None,
                dataset=dataset,
                box_alpha=0.3,
                show_class=True,
                thresh=0.7,
                kp_thresh=2
            )
    print('\n'.join(['%-30s %6.4fs' % (k, t.average_time) for k, t in timers.items()]))


if __name__ == '__main__':
    main()
