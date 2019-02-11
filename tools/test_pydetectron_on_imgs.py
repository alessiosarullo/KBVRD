import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch

from lib.pydetectron_api.detection import im_detect_all_with_feats
from lib.pydetectron_api.wrappers import \
    cfg, cfg_from_file, assert_and_infer_cfg, \
    segm_results, \
    dummy_datasets, \
    Generalized_RCNN, \
    misc_utils, vis_utils, Timer, load_detectron_weight, get_image_blob


def parse_args():
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')

    parser.add_argument('--cfg', dest='cfg_file', help='config file', required=True)
    parser.add_argument('--load_detectron', help='path to the detectron weight pickle file', required=True)
    parser.add_argument('--image_dir', help='directory to load images for demo', required=True)
    parser.add_argument('--num_imgs', help='how many images to perform detection on', default=8)
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
                 '--image_dir', 'data/HICO-DET/images/test2015',
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

    imglist = sorted(misc_utils.get_imagelist_from_dir(args.image_dir))[:args.num_imgs]
    num_images = len(imglist)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    timers = defaultdict(Timer)
    for i in range(num_images):
        print('img', i)
        im = cv2.imread(imglist[i])
        assert im is not None

        im_blob, _, im_info = get_image_blob(im, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE)
        inputs = {'data': torch.tensor(im_blob, device=device),
                  'im_info': torch.Tensor(im_info),
                  }
        box_class_scores, boxes, box_classes, im_ids, masks, feat_map, all_scores = im_detect_all_with_feats(maskRCNN, inputs, timers=timers)
        boxes_with_scores = np.concatenate([boxes, box_class_scores[:, None]], axis=1)
        cls_boxes = [[]] + [boxes_with_scores[box_classes == j, :] for j in range(1, cfg.MODEL.NUM_CLASSES)]

        timers['device_transfer'].tic()
        masks = masks.cpu().numpy()
        timers['device_transfer'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(cls_boxes, masks, boxes, im.shape[0], im.shape[1])
        timers['misc_mask'].toc()

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
