import argparse

import numpy as np
import numpy.random as npr
import torch
import torch.nn as nn

from lib.drivers.datasets import HicoDetSplit
from lib.utils.data import Minibatch
from config import Configs as cfg
from .mask_rcnn import MaskRCNN


class BaseModel(nn.Module):
    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__()

        self.filter_rels_of_non_overlapping_boxes = True  # FIXME params
        self.__dict__.update({k: v for k, v in vars(kwargs).items() if k in self.__dict__.keys() and v is not None})

        self.dataset = dataset
        self.mask_rcnn = MaskRCNN(model_name=cfg.model.rcnn_arch, num_classes=dataset.num_object_classes + 1)  # add BG class

    def forward(self, x, **kwargs):
        with torch.set_grad_enabled(self.training):
            return self._forward(x, **kwargs)

    def _forward(self, x: Minibatch, **kwargs):
        """
        :param x:
        :param kwargs:
        :return:
        """
        # TODO docs
        boxes_ext, box_feats, masks, rel_labels_ext_np, rel_feats = self.first_step(x)
        # TODO. This is all torch now!

    def first_step(self, x: Minibatch, **kwargs):  # FIXME change name
        """
        :param x:
        :param kwargs:
        :return:
        """
        # TODO docs
        # FIXME too much numpy

        boxes_ext_np, masks, feat_map, box_feats = self.mask_rcnn(x)
        # `boxes_ext_np` is Bx(1+4+C) where each row is [im_id, bbox_coord, class_scores]. Classes are COCO ones.

        boxes_ext_np, masks = self.filter_and_map_to_hico(boxes_ext_np, masks)

        gt_idx_per_pred_box, pred_gt_box_ious = iou_match_in_img(boxes_ext_np[:, :5], np.concatenate([x.gt_box_im_ids[:, None], x.gt_boxes], axis=1))
        # TODO add gt boxes to predicted boxes

        # TODO merge this with get_rel_inds?
        rel_labels_ext_np = rel_assignments(boxes_ext_np, x.gt_boxes, x.gt_box_im_ids, x.gt_box_classes, x.gt_inters, x.gt_inters_im_ids)
        # `rel_labels_ext_np` is Rx4 where each row is [im_id, sub_ind, pred, obj_ind]. Note that box indices are over all boxes, NOT relative to
        # each specific image

        assert rel_labels_ext_np.shape[0] > 0  # FIXME this is just a reminder to deal with that case, it's not actually true

        if self.training:
            ho_pairs = rel_labels_ext_np[:, [1, 3]]
        else:
            ho_pairs = self.get_all_pairs(boxes_ext_np)
        rel_union_boxes = self.get_union_boxes(boxes_ext_np[:, 1:5], ho_pairs)
        rel_feats = self.mask_rcnn(None, head_only=True, fmap=feat_map, rois=rel_union_boxes)  # FIXME is this horrible? Maybe don't use forward?
        assert rel_labels_ext_np.shape[0] == rel_feats.shape[0]

        boxes_ext = torch.tensor(boxes_ext_np, device=masks.device)
        rel_labels_ext = torch.tensor(rel_labels_ext_np, device=masks.device)
        return boxes_ext, box_feats, masks, rel_labels_ext, rel_feats

    def filter_and_map_to_hico(self, boxes_ext, masks):
        class_scores = boxes_ext[:, 5:]
        classes = np.argmax(class_scores, axis=1)
        fg_inds = (classes > 0)
        assert fg_inds.shape[0] == boxes_ext.shape[0] == masks.shape[0]
        boxes_ext, masks = boxes_ext[fg_inds, :], masks[fg_inds, :]
        boxes_ext = boxes_ext[:, list(range(5)) + self.dataset.hicodet.map_coco_classes_to_hico()]  # convert to Hico classes by swapping columns
        return boxes_ext, masks

    def get_all_pairs(self, boxes_ext, box_classes=None):
        box_classes = box_classes or np.argmax(boxes_ext[5:], axis=1)
        person_box_inds = (box_classes == self.dataset.hicodet.person_class)
        person_boxes_ext = boxes_ext[person_box_inds, :]

        if self.filter_rels_of_non_overlapping_boxes:
            _, pred_box_ious = iou_match_in_img(person_boxes_ext[:, :5], boxes_ext[:, :5])
            possible_rels_mat = 0 < pred_box_ious < 1
            subjs, objs = np.where(possible_rels_mat)
            subjs = np.flatnonzero(person_box_inds)[subjs]
        else:
            block_img_mat = (boxes_ext[:, 0][:, None] == boxes_ext[:, 0][None, :])
            assert block_img_mat.shape[0] == block_img_mat.shape[1]
            possible_rels_mat = block_img_mat - np.eye(block_img_mat.shape[0])
            possible_rels_mat = possible_rels_mat[person_box_inds, :]
            subjs, objs = np.where(possible_rels_mat)
        rel_inds = np.stack([subjs, objs], axis=1)  # this is over the original boxes, not person ones
        return rel_inds

    def get_union_boxes(self, boxes, union_inds):
        assert union_inds.shape[1] == 2
        union_rois = np.concatenate([
            np.minimum(boxes[:, :2][union_inds[:, 0]], boxes[:, :2][union_inds[:, 1]]),
            np.maximum(boxes[:, 2:][union_inds[:, 0]], boxes[:, 2:][union_inds[:, 1]]),
        ], axis=1)
        return union_rois


# TODO check and possibly update
def rel_assignments(boxes_ext_np, gt_boxes, gt_box_im_ids, gt_box_classes, gt_inters, gt_inters_im_ids,
                    num_sample_per_gt=4, filter_non_overlap=True, fg_rels_per_image=16):
    # FIXME enforce human subject only
    pred_box_im_ids = boxes_ext_np[:, 0]
    pred_boxes = boxes_ext_np[:, 1:5]
    pred_box_classes = np.argmax(boxes_ext_np[5:], axis=1)

    rel_labels_ext = []
    num_box_seen = 0
    for im_id in np.unique(gt_box_im_ids):
        pred_box_im_id = (pred_box_im_ids == im_id)
        gt_box_im_id = (gt_box_im_ids == im_id)

        gt_boxes_i = gt_boxes[gt_box_im_id]
        gt_classes_i = gt_box_classes[gt_box_im_id]
        gt_rels_i = gt_inters[gt_inters_im_ids == im_id]

        # [num_pred, num_gt]
        pred_boxes_i = pred_boxes[pred_box_im_id]
        pred_boxlabels_i = pred_box_classes[pred_box_im_id]

        ious = bbox_overlaps(pred_boxes_i, gt_boxes_i)
        is_match = (pred_boxlabels_i[:, None] == gt_classes_i[None]) & (ious >= 0.5)  # FIXME magic constant

        # FOR BG. Limit ourselves to only IOUs that overlap, but are not the exact same box
        pbi_iou = bbox_overlaps(pred_boxes_i, pred_boxes_i)
        if filter_non_overlap:
            rel_possibilities = (pbi_iou < 1) & (pbi_iou > 0)
            rels_intersect = rel_possibilities
        else:
            rel_possibilities = np.ones((pred_boxes_i.shape[0], pred_boxes_i.shape[0]),
                                        dtype=np.int64) - np.eye(pred_boxes_i.shape[0],
                                                                 dtype=np.int64)
            rels_intersect = (pbi_iou < 1) & (pbi_iou > 0)

        # ONLY select relations between ground truth because otherwise we get useless data
        rel_possibilities[pred_boxlabels_i == 0] = 0
        rel_possibilities[:, pred_boxlabels_i == 0] = 0

        # Sample the GT relationships.
        fg_rels = []
        p_size = []
        for i, (from_gtind, rel_id, to_gtind) in enumerate(gt_rels_i):
            fg_rels_i = []
            fg_scores_i = []

            for from_ind in np.where(is_match[:, from_gtind])[0]:
                for to_ind in np.where(is_match[:, to_gtind])[0]:
                    if from_ind != to_ind:
                        fg_rels_i.append((from_ind, to_ind, rel_id))
                        fg_scores_i.append((ious[from_ind, from_gtind] * ious[to_ind, to_gtind]))
                        rel_possibilities[from_ind, to_ind] = 0
            if len(fg_rels_i) == 0:
                continue
            p = np.array(fg_scores_i)
            p = p / p.sum()
            p_size.append(p.shape[0])
            num_to_add = min(p.shape[0], num_sample_per_gt)
            for rel_to_add in npr.choice(p.shape[0], p=p, size=num_to_add, replace=False):
                fg_rels.append(fg_rels_i[rel_to_add])

        fg_rels = np.array(fg_rels, dtype=np.int64)
        if fg_rels.size > 0 and fg_rels.shape[0] > fg_rels_per_image:
            fg_rels = fg_rels[npr.choice(fg_rels.shape[0], size=fg_rels_per_image, replace=False)]
        elif fg_rels.size == 0:
            fg_rels = np.zeros((0, 3), dtype=np.int64)

        bg_rels = np.column_stack(np.where(rel_possibilities))
        bg_rels = np.column_stack((bg_rels, np.zeros(bg_rels.shape[0], dtype=np.int64)))

        num_bg_rel = min(64 - fg_rels.shape[0], bg_rels.shape[0])
        if bg_rels.size > 0:
            bg_rels = bg_rels[np.random.choice(bg_rels.shape[0], size=num_bg_rel, replace=False)]
        else:
            bg_rels = np.zeros((0, 3), dtype=np.int64)

        if fg_rels.size == 0 and bg_rels.size == 0:
            # Just put something here
            bg_rels = np.array([[0, 0, 0]], dtype=np.int64)

        # print("GTR {} -> AR {} vs {}".format(gt_rels.shape, fg_rels.shape, bg_rels.shape))
        all_rels_i = np.concatenate((fg_rels, bg_rels), 0)
        all_rels_i[:, 0:2] += num_box_seen

        all_rels_i = all_rels_i[np.lexsort((all_rels_i[:, 1], all_rels_i[:, 0]))]

        rel_labels_ext.append(np.column_stack([
            im_id * np.ones(all_rels_i.shape[0], dtype=np.int64),
            all_rels_i,
        ]))

        num_box_seen += pred_boxes_i.shape[0]
    rel_labels_ext = np.concatenate(rel_labels_ext, axis=0)
    rel_labels_ext = rel_labels_ext[:, [0, 1, 3, 2]]  # [im_id, sub_ind, pred, obj_ind]
    return rel_labels_ext


def iou_match_in_img(boxes1, boxes2):
    box_im_ids1 = boxes1[:, 0]
    box_im_ids2 = boxes2[:, 0]
    ious = bbox_overlaps(boxes1[:, 1:5], boxes2[:, 1:5])
    ious[box_im_ids1[:, None] != box_im_ids2[None, :]] = 0.0
    # max_ious, argmax_ious = ious.max(dim=1)  # FIXME keep or delete
    argmax_ious = np.argmax(ious, axis=1)
    return argmax_ious, ious


# TODO check and possibly update
def bbox_overlaps(boxes_a, boxes_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        boxes_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        boxes_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    # FIXME a lot of duplication. Also docs
    if isinstance(boxes_a, np.ndarray):
        assert isinstance(boxes_b, np.ndarray)
        max_xy = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
        min_xy = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
        intersection_dims = np.maximum(0, max_xy - min_xy + 1.0)  # A x B x 2, where last dim is [width, height]
        intersections_areas = intersection_dims[:, :, 0] * intersection_dims[:, :, 1]

        areas_a = ((boxes_a[:, 2] - boxes_a[:, 0] + 1.0) *
                   (boxes_a[:, 3] - boxes_a[:, 1] + 1.0))[:, None]  # Ax1
        areas_b = ((boxes_b[:, 2] - boxes_b[:, 0] + 1.0) *
                   (boxes_b[:, 3] - boxes_b[:, 1] + 1.0))[None, :]  # 1xB
        union_areas = areas_a + areas_b - intersections_areas
        return intersections_areas / union_areas
    else:
        A = boxes_a.size(0)
        B = boxes_b.size(0)
        max_xy = torch.min(boxes_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                           boxes_b[:, 2:].unsqueeze(0).expand(A, B, 2))
        min_xy = torch.max(boxes_a[:, :2].unsqueeze(1).expand(A, B, 2),
                           boxes_b[:, :2].unsqueeze(0).expand(A, B, 2))
        inter = torch.clamp((max_xy - min_xy + 1.0), min=0)
        inter = inter[:, :, 0] * inter[:, :, 1]
        area_a = ((boxes_a[:, 2] - boxes_a[:, 0] + 1.0) *
                  (boxes_a[:, 3] - boxes_a[:, 1] + 1.0)).unsqueeze(1).expand_as(inter)  # [A,B]
        area_b = ((boxes_b[:, 2] - boxes_b[:, 0] + 1.0) *
                  (boxes_b[:, 3] - boxes_b[:, 1] + 1.0)).unsqueeze(0).expand_as(inter)  # [A,B]
        union = area_a + area_b - inter
        return inter / union  # [A,B]


def main():
    pass


if __name__ == '__main__':
    main()
