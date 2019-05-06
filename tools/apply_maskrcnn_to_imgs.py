import argparse
import torch
import sys
import os
import os.path as osp
from collections import defaultdict
import cv2

sys.path.insert(0, osp.abspath(osp.join('pydetectron', 'lib')))

# Note: fixing these imports doesn't work if Detectron's ones stay the same, because they end up looking at different configs (for some reason).
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as dummy_datasets
import utils.misc as misc_utils
import utils.vis as vis_utils
import utils.net as net_utils
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer
sys.path.remove(osp.abspath(osp.join('pydetectron', 'lib')))


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

    # model = 'e2e_mask_rcnn_R-50-FPN_1x'
    model = 'e2e_mask_rcnn_R-50-C4_2x'
    sys.argv += ['--cfg', 'pydetectron/configs/baselines/%s.yaml' % model,
                 '--load_detectron', 'data/pretrained_model/%s.pkl' % model,
                 '--image_dir', 'data/HICO-DET/images/train2015',
                 '--num_imgs', 0,
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

    # if args.load_detectron:
    #     load_name = args.load_detectron
    #     print("loading checkpoint %s" % (load_name))
    #     checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
    #     net_utils.load_ckpt(maskRCNN, checkpoint['model'])
    if args.load_detectron:
        print("loading detectron weights %s" % args.load_detectron)
        load_detectron_weight(maskRCNN, args.load_detectron)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'], minibatch=True, device_ids=[0])  # only support single GPU
    maskRCNN.eval()

    imglist = sorted(misc_utils.get_imagelist_from_dir(args.image_dir))
    if args.num_imgs != 0:
        imglist = imglist[:args.num_imgs]
    num_images = len(imglist)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    timers = defaultdict(Timer)
    for i in range(num_images):
        print('img', i)
        if not imglist[i].endswith('HICO_train2015_00001418.jpg'):
            continue
        im = cv2.imread(imglist[i])
        assert im is not None

        cls_boxes, cls_segms, cls_keyps = im_detect_all(maskRCNN, im, timers=timers)

        im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
        vis_utils.vis_one_image(
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            args.output_dir,
            cls_boxes,
            cls_segms,
            cls_keyps,
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.7,
            kp_thresh=2
        )
    print('\n'.join(['%20s: %6.4fs' % (k, t.average_time) for k, t in timers.items()]))


if __name__ == '__main__':
    main()
