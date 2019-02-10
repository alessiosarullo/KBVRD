import argparse
import os
import sys
from collections import defaultdict

import cv2
import numpy as np
import torch
import torch.nn as nn

from lib.drivers.datasets import HicoDetSplit, Splits, Minibatch
from lib.pydetectron_api.detection import im_detect_all_with_feats
from lib.pydetectron_api.wrappers import \
    cfg, cfg_from_file, assert_and_infer_cfg, \
    segm_results, \
    dummy_datasets, \
    Generalized_RCNN, \
    misc_utils, vis_utils, Timer, load_detectron_weight


class MaskRCNN(nn.Module):
    """
    Wrapper around Detectron's Mask-RCNN
    """

    def __init__(self, model_name, num_classes):
        super().__init__()

        if not torch.cuda.is_available():
            raise ValueError("Need a CUDA device to run the code.")  # TODO check if this is true

        cfg_file = 'pydetectron/configs/baselines/%s.yaml' % model_name
        weight_file = 'data/pretrained_model/%s.pkl' % model_name

        print("Loading Detectron's configs from {}.".format(cfg_file))
        cfg_from_file(cfg_file)
        cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
        cfg.MODEL.NUM_CLASSES = num_classes
        assert_and_infer_cfg()

        mask_rcnn = Generalized_RCNN()
        print("Loading Mask-RCNN's weights from {}.".format(weight_file))
        load_detectron_weight(mask_rcnn, weight_file)

        # self.device = torch.device('cuda')  # FIXME is this needed?
        self.cfg = cfg
        self.mask_rcnn = mask_rcnn

    def forward(self, x, **kwargs):
        with torch.set_grad_enabled(self.training):
            return self._forward(x, **kwargs)

    def _forward(self, x: Minibatch, **kwargs):
        """
        :param x:
        :param kwargs:
        :return: NOTE: classes here include BG one, not present in HICO
        """
        # TODO docs
        # TODO coco-hico class mapping

        assert not self.training
        batch_tensor = x.imgs
        orig_im_infos = x.img_infos

        im_scales = orig_im_infos[:, 2]
        im_infos = np.concatenate([np.tile(batch_tensor.shape[2:], reps=[im_scales.size, 1]), im_scales[:, None]], axis=1)
        inputs = {'data': batch_tensor,
                  'im_info': torch.Tensor(im_infos), }
        print(orig_im_infos)
        print(im_infos)
        scores, boxes, box_classes, im_ids, masks, feat_map = im_detect_all_with_feats(self.mask_rcnn, inputs, timers=None)
        return scores, boxes, box_classes, im_ids, masks, feat_map


def main():
    output_dir = 'detectron_outputs/test_ds/'

    batch_size = 2
    num_images = 8

    hds = HicoDetSplit(Splits.TEST, im_inds=list(range(num_images)))
    hdsl = hds.get_loader(batch_size=batch_size)
    dummy_coco = dummy_datasets.get_coco_dataset()  # this is used for class names

    mask_rcnn = MaskRCNN(model_name='e2e_mask_rcnn_R-50-C4_2x', num_classes=hds.num_object_classes + 1)  # add BG class
    mask_rcnn.cuda()
    mask_rcnn.eval()

    for batch_i, batch in enumerate(hdsl):
        print('Batch', batch_i)
        batch = batch  # type: Minibatch

        scores, boxes, box_classes, im_ids, masks, feat_map = mask_rcnn(batch)

        boxes_with_scores = np.concatenate([boxes, scores[:, None]], axis=1)
        masks = masks.cpu().numpy()
        for i in np.unique(im_ids):
            binmask_i = (im_ids == i)
            boxes_with_scores_i = boxes_with_scores[binmask_i, :]
            box_classes_i = box_classes[binmask_i]
            masks_i = masks[binmask_i]

            im_fn = batch.img_fns[i]
            im = cv2.imread(os.path.join(hds.img_dir, im_fn))

            cls_boxes = [[]] + [boxes_with_scores_i[box_classes_i == j, :] for j in range(1, cfg.MODEL.NUM_CLASSES)]
            cls_segms = segm_results(cls_boxes, masks_i, boxes_with_scores_i[:, :-1], im.shape[0], im.shape[1])

            vis_utils.vis_one_image(
                im[:, :, [2, 1, 0]],  # BGR -> RGB for visualization
                os.path.splitext(batch.img_fns[i])[0],
                output_dir,
                cls_boxes,
                cls_segms,
                None,
                dataset=dummy_coco,
                box_alpha=0.3,
                show_class=True,
                thresh=0.7,
                kp_thresh=2
            )

    import pickle
    with open('cache/tmp_mask_result.pkl', 'wb') as f:
        pickle.dump((scores, boxes, box_classes, im_ids, feat_map), f)


if __name__ == '__main__':
    main()
