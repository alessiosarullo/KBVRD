import argparse

import numpy as np
import torch
import torch.nn as nn

from lib.drivers.datasets import HicoDetSplit
from .mask_rcnn import MaskRCNN


class BaseModel(nn.Module):
    def __init__(self, dataset: HicoDetSplit, mask_args):
        super().__init__()

        self.dataset = dataset
        self.maskRCNN = MaskRCNN()

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

    def _forward(self, x, **kwargs):
        fmaps, boxes_list, masks_list = self.maskRCNN(x)
        obj_labels = self.assign_labels_to_predicted_boxes(boxes_list, x['gt_boxes'], x['gt_box_classes'])


    @staticmethod
    def assign_labels_to_predicted_boxes(predicted_boxes, gt_boxes, gt_box_classes):
        all_boxes_rcnn = []
        im_inds_rcnn = []
        for im_ind, rcnn_boxes in enumerate(predicted_boxes):
            all_boxes_rcnn.append(rcnn_boxes)
            im_inds_rcnn.append(np.ones(rcnn_boxes.shape[0]) * im_ind)
        all_boxes_rcnn = torch.cat(all_boxes_rcnn, dim=0)
        im_inds_rcnn = np.concatenate(im_inds_rcnn)

        all_boxes_gt = []
        all_labels_gt = []
        im_inds_gt = []
        for im_ind, (gt_boxes, gt_classes) in enumerate(zip(gt_boxes, gt_box_classes)):
            all_boxes_gt.append(gt_boxes)
            all_labels_gt.append(gt_classes)
            im_inds_gt.append(np.ones(gt_boxes.shape[0]) * im_ind)
        all_boxes_gt = torch.cat(all_boxes_gt, dim=0)
        all_labels_gt = torch.cat(all_labels_gt, dim=0)
        im_inds_gt = np.concatenate(im_inds_gt)

        pred_to_gtbox = bbox_overlaps(all_boxes_rcnn, all_boxes_gt)
        pred_to_gtbox[im_inds_rcnn[:, None] != im_inds_gt[None, :]] = 0.0

        max_overlaps, argmax_overlaps = pred_to_gtbox.max(dim=1)
        obj_labels = all_labels_gt[argmax_overlaps]
        obj_labels[max_overlaps < 0.5] = 0

        return obj_labels



#TODO check these and possibly update them
def bbox_overlaps(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    if isinstance(box_a, np.ndarray):
        assert isinstance(box_b, np.ndarray)
        assert False  # FIXME this should not happen (boxes should be PyTorch tensors)
        return bbox_overlaps_np(box_a, box_b)

    inter = bbox_intersections(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0] + 1.0) *
              (box_a[:, 3] - box_a[:, 1] + 1.0)).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2] - box_b[:, 0] + 1.0) *
              (box_b[:, 3] - box_b[:, 1] + 1.0)).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def bbox_intersections(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy + 1.0), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


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
