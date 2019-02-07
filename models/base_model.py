import argparse
import sys
from collections import defaultdict

import numpy as np
import pycocotools.mask as mask_util
import torch
import torch.nn as nn
import torch.nn.parallel

from drivers.datasets import HicoDetSplit
from models.pydetectron.lib.core.config import cfg
from models.pydetectron.lib.core.test_engine import initialize_model_from_cfg
from models.pydetectron.lib.utils.timer import Timer
from models.pydetectron.lib.utils.vis import convert_from_cls_format


class BaseModel(nn.Module):
    def __init__(self, dataset: HicoDetSplit, mask_args):
        super().__init__()

        self.dataset = dataset
        self.maskRCNN = initialize_model_from_cfg(mask_args, gpu_id=mask_args.gpu_id)

    def forward(self, x, **kwargs):
        """
        :param x:  {'imgs': [],
                    'img_size': [],
                    'gt_boxes': [],
                    'gt_box_classes': [],
                    'gt_inters': [],
                    }
        :return:
        """

        with torch.set_grad_enabled(self.training):
            return self._forward(x, **kwargs)

    def im_detect_all(self, im, img_info):
        model = self.maskRCNN
        timers = defaultdict(Timer)

        timers['im_detect_bbox'].tic()
        assert not cfg.TEST.BBOX_AUG.ENABLED
        scores, boxes, blob_conv = self.im_detect_bbox(model, im, img_info)
        timers['im_detect_bbox'].toc()

        # score and boxes are from the whole image after score thresholding and nms(they are not separated by class) (numpy.ndarray)
        # cls_boxes boxes and scores are separated by class and in the format used for evaluating results
        timers['misc_bbox'].tic()
        scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
        timers['misc_bbox'].toc()

        assert cfg.MODEL.MASK_ON and boxes.shape[0] > 0
        timers['im_detect_mask'].tic()
        if cfg.TEST.MASK_AUG.ENABLED:
            masks = im_detect_mask_aug(model, im, boxes, im_scale, blob_conv)
        else:
            masks = im_detect_mask(model, im_scale, boxes, blob_conv)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(cls_boxes, masks, boxes, im.shape[0], im.shape[1])
        timers['misc_mask'].toc()

        assert not cfg.MODEL.KEYPOINTS_ON

        return cls_boxes, cls_segms, blob_conv

    def im_detect_bbox(self, model, im, img_info):
        assert cfg.MODEL.FASTER_RCNN
        assert not cfg.PYTORCH_VERSION_LESS_THAN_040
        assert cfg.TEST.BBOX_REG
        assert not cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED
        assert not cfg.MODEL.CLS_AGNOSTIC_BBOX_REG

        inputs = {'data': [im],
                  'im_info': [img_info]}

        model_output = model(**inputs)

        boxes = model_output['rois'][:, 1:5]

        scores = model_output['cls_score']  # cls prob (activations after softmax)
        # scores = scores.reshape([-1, scores.shape[-1]])  # In case there is 1 proposal

        box_deltas = model_output['bbox_pred']
        # box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])  # In case there is 1 proposal
        pred_boxes = box_utils.bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im.shape)

        return scores, pred_boxes, model_output['blob_conv']

    def _forward(self, x, **kwargs):
        boxes_list, masks_list, img_inds, fmaps = [], [], [], []
        for im_i, (im, img_info) in enumerate(zip(x['img'], x['img_info'])):
            cls_boxes_i, cls_segms_i, fmap = self.im_detect_all(im, img_info)
            boxes_i, segms_i, keyps_i, classes_i = convert_from_cls_format(cls_boxes_i, cls_segms_i, None)

            assert boxes_i is not None and segms_i is not None and boxes_i.shape[0] == segms_i.shape[0].shape[0] > 0
            masks_i = mask_util.decode(segms_i)

            boxes_list.append(boxes_i)
            masks_list.append(masks_i)
            img_inds += [im_i] * boxes_i.shape[0]
            fmaps.append(fmap)

        boxes_w_scores = np.concatenate(boxes_list, axis=0)
        masks = np.concatenate(masks_list, axis=0)
        img_inds = np.array(img_inds)
        fmaps = np.stack(fmaps, axis=0)

        self.proposal_assignments_det(boxes)

    def proposal_assignments_det(self, rpn_rois, gt_boxes, gt_classes, fg_thresh=0.5):
        """
        Assign object detection proposals to ground-truth targets. Produces proposal
        classification labels and bounding-box regression targets.
        :param rpn_rois: [img_ind, x1, y1, x2, y2]
        :param gt_boxes:   [num_boxes, 4] array of x0, y0, x1, y1
        :param gt_classes: [num_boxes, 2] array of [img_ind, class]
        :param Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
        :return:
            rois: [num_rois, 5]
            labels: [num_rois] array of labels
            bbox_targets [num_rois, 4] array of targets for the labels.
        """
        fg_rois_per_image = int(np.round(ROIS_PER_IMG * FG_FRACTION))

        gt_img_inds = gt_classes[:, 0]

        all_boxes = torch.cat([rpn_rois[:, 1:], gt_boxes], dim=0)

        ims_per_box = torch.cat([rpn_rois[:, 0].long(), gt_img_inds], dim=0)

        im_sorted, idx = torch.sort(ims_per_box, 0)
        all_boxes = all_boxes[idx]

        # Assume that the GT boxes are already sorted in terms of image id
        num_images = int(im_sorted[-1]) + 1

        labels = []
        rois = []
        bbox_targets = []
        for im_ind in range(num_images):
            g_inds = (gt_img_inds == im_ind).nonzero()

            if g_inds.dim() == 0:
                continue
            g_inds = g_inds.squeeze(1)
            g_start = g_inds[0]
            g_end = g_inds[-1] + 1

            t_inds = (im_sorted == im_ind).nonzero().squeeze(1)
            t_start = t_inds[0]
            t_end = t_inds[-1] + 1

            # Max overlaps: for each predicted box, get the max ROI
            # Get the indices into the GT boxes too (must offset by the box start)
            ious = bbox_overlaps(all_boxes[t_start:t_end], gt_boxes[g_start:g_end])
            max_overlaps, gt_assignment = ious.max(1)
            max_overlaps = max_overlaps.cpu().numpy()
            # print("Best overlap is {}".format(max_overlaps.max()))
            # print("\ngt assignment is {} while g_start is {} \n ---".format(gt_assignment, g_start))
            gt_assignment += g_start

            keep_inds_np, num_fg = _sel_inds(max_overlaps, fg_thresh, fg_rois_per_image,
                                             ROIS_PER_IMG)

            if keep_inds_np.size == 0:
                continue

            keep_inds = torch.LongTensor(keep_inds_np).cuda(rpn_rois.get_device())

            labels_ = gt_classes[:, 1][gt_assignment[keep_inds]]
            bbox_target_ = gt_boxes[gt_assignment[keep_inds]]

            # Clamp labels_ for the background RoIs to 0
            if num_fg < labels_.size(0):
                labels_[num_fg:] = 0

            rois_ = torch.cat((
                im_sorted[t_start:t_end, None][keep_inds].float(),
                all_boxes[t_start:t_end][keep_inds],
            ), 1)

            labels.append(labels_)
            rois.append(rois_)
            bbox_targets.append(bbox_target_)

        rois = torch.cat(rois, 0)
        labels = torch.cat(labels, 0)
        bbox_targets = torch.cat(bbox_targets, 0)
        return rois, labels, bbox_targets


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', dest='cfg_file', required=True, help='config file')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    # parser.add_argument('--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument('--set', dest='set_cfgs', help='Set config keys, overwriting config in the cfg_file. See lib/core/config.py for all options',
                        default=[], nargs='*')

    return parser.parse_args()


def main():
    pass


if __name__ == '__main__':
    main()
