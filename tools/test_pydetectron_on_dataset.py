import argparse
import os
import os.path as osp
import sys
from collections import defaultdict

import torch

from lib.drivers.datasets import SquarePad, Resize, ToTensor, Compose, im_list_to_4d_tensor
from lib.pydetectron_api.detection import im_detect_all_with_feats

sys.path.insert(0, osp.abspath(osp.join('pydetectron', 'lib')))
# Note: fixing these imports doesn't work if Detectron's ones stay the same, because they end up looking at different configs (for some reason).
# Might be because when imported using `from` not everything is imported
import nn as mynn
from core.config import cfg, cfg_from_file, assert_and_infer_cfg
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as dummy_datasets
import utils.misc as misc_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer
from core.test import segm_results
import utils.vis as vis_utils
from PIL import Image
sys.path.remove(osp.abspath(osp.join('pydetectron', 'lib')))


def get_transform_pipeline():
    tform = [
        SquarePad(),
        Resize(600),  # TODO move it so that the rescaling can be capped at a maximum value? (Probably not)
        ToTensor(),
        # Normalize(mean=cfg.data.pixel_mean, std=cfg.data.pixel_std),
    ]
    return Compose(tform)


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

    # model = 'e2e_mask_rcnn_R-50-FPN_1x'
    model = 'e2e_mask_rcnn_R-50-C4_2x'
    sys.argv += ['--cfg', 'pydetectron/configs/baselines/%s.yaml' % model,
                 '--load_detectron', 'data/pretrained_model/%s.pkl' % model,
                 '--image_dir', '/media/alex/Woodo/PhD/KB-HOI data/fake_HICO-DET',
                 '--output_dir', 'detectron_outputs/test/',
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

    vis = False
    batch_size = 1 if vis else 4
    batch_size = 1
    transform_pipeline = get_transform_pipeline()

    num_images = (num_images // batch_size) * batch_size
    batches, batch = [], []
    for i in range(num_images):
        print('img', i)
        im = Image.open(imglist[i]).convert('RGB')
        assert im is not None
        im = transform_pipeline(im)
        batch.append(im)
        if (i + 1) % batch_size == 0:
            batches.append(im_list_to_4d_tensor(batch))
            batch = []

    timers = defaultdict(Timer)
    for i, batch in enumerate(batches):
        print('Batch', i)
        assert len(batch) == 1
        im = batch[0]

        scores, boxes, masks, feat_map, cls_boxes = im_detect_all_with_feats(maskRCNN, im, timers=timers)

        timers['device_transfer'].tic()
        scores = scores.cpu().numpy()
        boxes = boxes.cpu().numpy()
        masks = masks.cpu().numpy()
        timers['device_transfer'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(cls_boxes, masks, boxes, im.shape[0], im.shape[1])
        timers['misc_mask'].toc()

        if vis:
            im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
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
