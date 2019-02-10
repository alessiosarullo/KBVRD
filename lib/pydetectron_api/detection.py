from collections import defaultdict

import numpy as np
import torch

from .wrappers import cfg, Timer, _get_rois_blob, _add_multilevel_rois_for_test, box_utils


def im_detect_all_with_feats(model, inputs, box_proposals=None, timers=None):
    """
    Returned `scores`, `boxes`, `box_classes`, `im_ids` are NumPy, `masks` and `feat_map` are Torch
    """

    assert not cfg.TEST.BBOX_AUG.ENABLED
    assert not cfg.MODEL.KEYPOINTS_ON
    assert cfg.MODEL.MASK_ON
    assert not cfg.TEST.MASK_AUG.ENABLED

    if timers is None:
        timers = defaultdict(Timer)

    im_scales = inputs['im_info'][:, 2].cpu().numpy()
    if box_proposals is not None:
        assert False  # FIXME did not check if this works
        inputs['rois'] = _get_rois_blob(box_proposals, [ims for ims in im_scales])

    timers['im_detect_bbox'].tic()
    nonnms_scores, nonnms_boxes, feat_map, nonnms_im_ids = _im_detect_bbox(model, inputs, im_scales, timers)
    timers['im_detect_bbox'].toc()
    assert nonnms_boxes.shape[0] > 0

    timers['device_transfer'].tic()
    nonnms_scores = nonnms_scores.cpu().numpy()
    nonnms_boxes = nonnms_boxes.cpu().numpy()
    timers['device_transfer'].toc()

    timers['bbox_nms'].tic()
    box_inds, box_classes, scores, boxes = _box_results_with_nms_and_limit(nonnms_scores, nonnms_boxes, nonnms_im_ids)
    im_ids = nonnms_im_ids[box_inds]
    timers['bbox_nms'].toc()

    assert boxes.shape[0] > 0
    # assert np.all(np.stack([all_boxes[i, j*4:(j+1)*4] for i, j in zip(box_inds, classes)], axis=0) == boxes)

    timers['im_detect_mask'].tic()
    masks = _im_detect_mask(model, im_scales, im_ids, boxes, feat_map)
    timers['im_detect_mask'].toc()

    return scores, boxes, box_classes, im_ids, masks, feat_map


def _im_detect_bbox(model, inputs, im_scales, timers=None):
    """Prepare the bbox for testing"""

    assert not cfg.PYTORCH_VERSION_LESS_THAN_040
    assert cfg.MODEL.FASTER_RCNN
    assert cfg.TEST.BBOX_REG
    assert not cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    assert not cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox - detect'].tic()
    return_dict = model(**inputs)
    box_deltas = return_dict['bbox_pred']
    scores = return_dict['cls_score']  # cls prob (activations after softmax)
    boxes = torch.tensor(return_dict['rois'][:, 1:5], device=box_deltas.device)
    im_inds = return_dict['rois'][:, 0].astype(np.int, copy=False)
    timers['im_detect_bbox - detect'].toc()

    factors = im_scales[im_inds]
    boxes /= boxes.new_tensor(factors[:, None])  # unscale back to raw image space
    scores = scores.view([-1, scores.shape[-1]])  # In case there is 1 proposal
    box_deltas = box_deltas.view([-1, box_deltas.shape[-1]])  # In case there is 1 proposal

    # Apply bounding-box regression deltas
    timers['im_detect_bbox - apply_delta'].tic()
    pred_boxes = _bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
    pred_boxes = _clip_tiled_boxes(pred_boxes, inputs['data'][0].shape[1:])
    timers['im_detect_bbox - apply_delta'].toc()

    return scores, pred_boxes, return_dict['blob_conv'], im_inds  # NOTE: pred_boxes are scaled back to the original image size


def _bbox_transform(boxes, deltas, weights):
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


def _clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width] and boxes has shape (N, 4 * num_tiled_boxes)."""
    assert boxes.shape[1] % 4 == 0, 'boxes.shape[1] is {:d}, but must be divisible by 4.'.format(boxes.shape[1])
    boxes[:, 0::4] = boxes[:, 0::4].clamp(min=0, max=im_shape[1] - 1)  # x1 >= 0
    boxes[:, 1::4] = boxes[:, 1::4].clamp(min=0, max=im_shape[0] - 1)  # y1 >= 0
    boxes[:, 2::4] = boxes[:, 2::4].clamp(min=0, max=im_shape[1] - 1)  # x2 < im_shape[1]
    boxes[:, 3::4] = boxes[:, 3::4].clamp(min=0, max=im_shape[0] - 1)  # y2 < im_shape[0]
    return boxes


def _box_results_with_nms_and_limit(all_scores, all_boxes, im_ids):
    """
    Returns bounding-box detection results by thresholding on scores and applying non-maximum suppression (NMS). In the following R denotes the
    number of detections and C the number of classes
    :param all_scores [array, R x C]: Each row represents a list of object detection confidence scores for each of the object classes in the dataset
                (including the background class). Element (i, j) corresponds to the box at `all_boxes[i, j * 4:(j + 1) * 4]`.
    :param all_boxes [array, R x 4C]: Each row represents a list of predicted bounding boxes for each of the object classes in  the dataset
                (including the background class). The detections in each row originate from the same object proposal.
    :param im_inds [array, R]: Which image the corresponding box belongs to.
    :return:
    """
    assert not cfg.TEST.SOFT_NMS.ENABLED
    assert not cfg.TEST.BBOX_VOTE.ENABLED

    num_classes = cfg.MODEL.NUM_CLASSES

    image_masks = [im_ids == im_id for im_id in np.unique(im_ids)]
    all_boxes_ids = np.arange(all_boxes.shape[0])

    # Apply threshold on detection probabilities and apply NMS. Skip j = 0, because it's the background class
    all_results = []
    for mask_i in image_masks:
        scores_i = all_scores[mask_i, :]
        boxes_i = all_boxes[mask_i, :]
        boxes_ids_i = all_boxes_ids[mask_i]

        boxes_and_infos_per_class = {}
        for j in range(1, num_classes):
            class_boxes_mask = scores_i[:, j] > cfg.TEST.SCORE_THRESH
            scores_ij = scores_i[class_boxes_mask, j]
            boxes_ij = boxes_i[class_boxes_mask, j * 4:(j + 1) * 4]
            boxes_ids_ij = boxes_ids_i[class_boxes_mask]

            keep = box_utils.nms(np.concatenate([boxes_ij, scores_ij[:, None]], axis=1).astype(np.float32, copy=False), cfg.TEST.NMS)

            scores_ij = scores_ij[keep]
            boxes_ids_ij = boxes_ids_ij[keep]
            boxes_ij = boxes_ij[keep, :]
            j_vec = np.ones_like(boxes_ids_ij) * j
            box_infos_ij = np.stack([scores_ij, boxes_ids_ij, j_vec], axis=1)
            boxes_and_infos_per_class[j] = np.concatenate([boxes_ij, box_infos_ij], axis=1)

        # Limit to max_per_image detections **over all classes**
        if cfg.TEST.DETECTIONS_PER_IM > 0:
            image_scores = np.concatenate([boxes_and_infos_per_class[j][:, 4] for j in range(1, num_classes)])
            if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
                image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
                for j in range(1, num_classes):
                    keep = boxes_and_infos_per_class[j][:, 4] >= image_thresh
                    boxes_and_infos_per_class[j] = boxes_and_infos_per_class[j][keep, :]

        im_result = np.concatenate([boxes_and_infos_per_class[j] for j in range(1, num_classes)], axis=0)
        all_results.append(im_result)

    all_results = np.concatenate(all_results, axis=0)
    boxes_ids = all_results[:, 5].astype(np.int)
    box_classes = all_results[:, 6].astype(np.int)
    scores = all_results[:, 4]
    boxes = all_results[:, :4]
    return boxes_ids, box_classes, scores, boxes


def _im_detect_mask(model, im_scales, im_inds, boxes, blob_conv):
    """
    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scale (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as returned by im_detect_bbox)
        blob_conv (Variable): base features from the backbone network.

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """

    assert cfg.MRCNN.CLS_SPECIFIC_MASK
    assert boxes.shape[0] > 0
    M = cfg.MRCNN.RESOLUTION

    boxes = boxes * im_scales[im_inds, None]
    mask_rois = np.concatenate([im_inds[:, None], boxes], axis=1).astype(np.float32, copy=False)

    inputs = {'mask_rois': mask_rois}
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'mask_rois')

    pred_masks = model.mask_net(blob_conv, inputs)
    pred_masks = pred_masks.view([-1, cfg.MODEL.NUM_CLASSES, M, M])
    return pred_masks
