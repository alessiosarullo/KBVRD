import torch
import numpy as np


def bbox_transform(boxes, deltas, weights, bbox_xform_clip):
    """
    Forward transform that maps proposal boxes to predicted ground-truth boxes using bounding-box regression deltas.
    """

    pred_boxes = deltas.new_zeros(deltas.shape)
    if boxes.shape[0] == 0:
        assert deltas.shape[0] == 0
        return pred_boxes
    assert boxes.dtype == deltas.dtype, (boxes.dtype, deltas.dtype)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    center_x = boxes[:, 0] + 0.5 * widths
    center_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into exp()
    dw.clamp_(max=bbox_xform_clip)
    dh.clamp_(max=bbox_xform_clip)

    pred_center_x = dx * widths[:, None] + center_x[:, None]
    pred_center_y = dy * heights[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes[:, 0::4] = pred_center_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1::4] = pred_center_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2::4] = pred_center_x + 0.5 * pred_w - 1  # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_center_y + 0.5 * pred_h - 1  # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)

    return pred_boxes


def clip_tiled_boxes(boxes, im_shapes):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 4 == 0, 'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(boxes.shape[1])
    boxes = boxes.clamp(min=0)
    boxes[:, 0::2] = torch.min(boxes[:, 0::2], im_shapes[:, [1]] - 1)
    boxes[:, 1::2] = torch.min(boxes[:, 1::2], im_shapes[:, [0]] - 1)
    # boxes[:, 3::4] = boxes[:, 3::4].clamp(max=im_shapes[:, 0] - 1)  # y2 < im_shape[0]
    return boxes
