import numpy as np

from .wrappers import cfg, box_utils, im_detect_bbox


def im_detect_boxes(model, img):
    assert not cfg.TEST.BBOX_AUG.ENABLED
    assert not cfg.MODEL.KEYPOINTS_ON
    assert not cfg.TEST.MASK_AUG.ENABLED

    nonnms_scores, nonnms_boxes, im_scale, feat_map = im_detect_bbox(model, img, cfg.TEST.SCALE, cfg.TEST.MAX_SIZE, box_proposals=None)
    assert nonnms_boxes.shape[0] > 0
    if isinstance(im_scale, list):
        assert len(im_scale) == 1
        im_scale = im_scale[0]
    # NOTE: at this point boxes are scaled back to the image size

    box_inds, boxes, _, box_scores = _box_results_with_nms_and_limit(nonnms_scores, nonnms_boxes)
    # Note: bounding boxes are regressed per class. Thus, same-object bounding boxes may differ and appear twice. We filter them out based on their
    # score, so that each detected bounding box actually belongs to a single object.
    s_inds = np.argsort(box_scores)[::-1]
    box_inds = box_inds[s_inds]
    boxes = boxes[s_inds]
    u_box_inds, u_idx = np.unique(box_inds, return_index=True)
    boxes = boxes[u_idx, :]

    scores = nonnms_scores[u_box_inds, :]

    # Filter out 0-area boxes
    nonzero_area_boxes = (boxes[:, 0] < boxes[:, 2]) & (boxes[:, 1] < boxes[:, 3])
    boxes = boxes[nonzero_area_boxes, :]
    scores = scores[nonzero_area_boxes, :]

    assert boxes.shape[0] == scores.shape[0]
    return boxes, scores, feat_map, im_scale


def _box_results_with_nms_and_limit(scores, boxes):
    """
    Returns bounding-box detection results by thresholding on scores and applying non-maximum suppression (NMS). In the following B denotes the
    number of detections and C the number of classes
    :param scores [array, B x C]: Each row represents a list of object detection confidence scores for each of the object classes in the dataset
                (including the background class). Element (i, j) corresponds to the box at `boxes[i, j * 4:(j + 1) * 4]`.
    :param boxes [array, B x 4C]: Each row represents a list of predicted bounding boxes for each of the object classes in  the dataset
                (including the background class). The detections in each row originate from the same object proposal.
    :return:
    """
    assert not cfg.TEST.SOFT_NMS.ENABLED
    assert not cfg.TEST.BBOX_VOTE.ENABLED

    num_classes = cfg.MODEL.NUM_CLASSES

    boxes_ids = np.arange(boxes.shape[0])

    # Apply threshold on detection probabilities and apply NMS. Skip j = 0, because it's the background class
    max_score = np.amax(scores[:, 1:])

    boxes_and_infos_per_class = {}
    for j in range(1, num_classes):
        class_boxes_mask = scores[:, j] >= min(cfg.TEST.SCORE_THRESH, max_score)
        scores_j = scores[class_boxes_mask, j]
        boxes_j = boxes[class_boxes_mask, j * 4:(j + 1) * 4]
        boxes_ids_j = boxes_ids[class_boxes_mask]

        rois_j = np.concatenate([boxes_j, scores_j[:, None]], axis=1).astype(np.float32, copy=False)
        keep = box_utils.nms(rois_j, cfg.TEST.NMS)
        # if np.any(class_boxes_mask):
        #     print(i, j, len(keep))

        scores_j = scores_j[keep]
        boxes_ids_j = boxes_ids_j[keep]
        boxes_j = boxes_j[keep, :]
        j_vec = np.full_like(boxes_ids_j, fill_value=j)
        box_infos_j = np.stack([scores_j, boxes_ids_j, j_vec], axis=1)
        boxes_and_infos_per_class[j] = np.concatenate([boxes_j, box_infos_j], axis=1)

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.concatenate([boxes_and_infos_per_class[j][:, 4] for j in range(1, num_classes)])
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = boxes_and_infos_per_class[j][:, 4] >= image_thresh
                boxes_and_infos_per_class[j] = boxes_and_infos_per_class[j][keep, :]

    results = np.concatenate([boxes_and_infos_per_class[j] for j in range(1, num_classes)], axis=0)
    boxes_ids = results[:, 5].astype(np.int)
    box_classes = results[:, 6].astype(np.int)
    scores = results[:, 4]
    boxes = results[:, :4].astype(np.float32, copy=False)
    return boxes_ids, boxes, box_classes, scores


def get_rois_feats(model, fmap, rois):
    # Input to Box_Head should be a dictionary with the field 'rois' as a Bx5 NumPy array, where each row is [im_id, x1, y1, x2, y2]
    rois_feats = model.Box_Head(fmap, {'rois': rois})
    assert all([s == 1 for s in rois_feats.shape[2:]])
    rois_feats.squeeze_(dim=3)
    rois_feats.squeeze_(dim=2)
    return rois_feats
