import torch

from lib.pydetectron_integration.wrappers import cfg


def bbox_transform(boxes, deltas, weights):
    """
    Forward transform that maps proposal boxes to predicted ground-truth boxes using bounding-box regression deltas.
    """

    pred_boxes = deltas.new_zeros(deltas.shape)
    if boxes.shape[0] == 0:
        assert deltas.shape[0] == 0
        return pred_boxes
    assert boxes.dtype == deltas.dtype

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
    dw.clamp_(max=cfg.BBOX_XFORM_CLIP)
    dh.clamp_(max=cfg.BBOX_XFORM_CLIP)

    pred_center_x = dx * widths[:, None] + center_x[:, None]
    pred_center_y = dy * heights[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * widths[:, None]
    pred_h = torch.exp(dh) * heights[:, None]

    pred_boxes[:, 0::4] = pred_center_x - 0.5 * pred_w  # x1
    pred_boxes[:, 1::4] = pred_center_y - 0.5 * pred_h  # y1
    pred_boxes[:, 2::4] = pred_center_x + 0.5 * pred_w - 1  # x2 (note: "- 1" is correct; don't be fooled by the asymmetry)
    pred_boxes[:, 3::4] = pred_center_y + 0.5 * pred_h - 1  # y2 (note: "- 1" is correct; don't be fooled by the asymmetry)

    return pred_boxes


def clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 4 == 0, 'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(boxes.shape[1])
    boxes[:, 0::4] = boxes[:, 0::4].clamp(min=0, max=im_shape[1] - 1)  # x1 >= 0
    boxes[:, 1::4] = boxes[:, 1::4].clamp(min=0, max=im_shape[0] - 1)  # y1 >= 0
    boxes[:, 2::4] = boxes[:, 2::4].clamp(min=0, max=im_shape[1] - 1)  # x2 < im_shape[1]
    boxes[:, 3::4] = boxes[:, 3::4].clamp(min=0, max=im_shape[0] - 1)  # y2 < im_shape[0]
    return boxes
