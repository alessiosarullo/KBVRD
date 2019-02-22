import numpy as np
import torch


def iou_match_in_img(boxes1, boxes2):
    box_im_ids1 = boxes1[:, 0]
    box_im_ids2 = boxes2[:, 0]
    ious = compute_ious(boxes1[:, 1:5], boxes2[:, 1:5])
    ious[box_im_ids1[:, None] != box_im_ids2[None, :]] = 0.0
    argmax_ious = np.argmax(ious, axis=1)
    return argmax_ious, ious


# TODO check and possibly update
def compute_ious(boxes_a, boxes_b):
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


def get_union_boxes(boxes, union_inds):
    assert union_inds.shape[1] == 2
    union_rois = np.concatenate([
        np.minimum(boxes[:, :2][union_inds[:, 0]], boxes[:, :2][union_inds[:, 1]]),
        np.maximum(boxes[:, 2:][union_inds[:, 0]], boxes[:, 2:][union_inds[:, 1]]),
    ], axis=1)
    return union_rois
