import numpy as np
import torch

from lib.pydetectron_integration.box_utils import bbox_transform, clip_tiled_boxes
from scripts.utils import Timer

from .wrappers import cfg, _add_multilevel_rois_for_test, box_utils, nms_gpu


def im_detect_all_with_feats(model, inputs):
    """
    Returned `scores`, `boxes`, `box_classes`, `im_ids` are NumPy, `masks` and `feat_map` are Torch
    """
    # TODO docs

    assert not cfg.TEST.BBOX_AUG.ENABLED
    assert not cfg.MODEL.KEYPOINTS_ON
    assert cfg.MODEL.MASK_ON
    assert not cfg.TEST.MASK_AUG.ENABLED

    # im_scales = inputs['im_info'][:, 2]
    im_scales_np = inputs['im_info'][:, 2].cpu().numpy()

    Timer.get('Epoch', 'Batch', 'Detect', 'ImDetBox').tic()
    nonnms_scores, nonnms_boxes, feat_map, nonnms_im_ids = _im_detect_bbox(model, inputs)
    Timer.get('Epoch', 'Batch', 'Detect', 'ImDetBox').toc()
    assert nonnms_boxes.shape[0] > 0

    nonnms_scores = nonnms_scores.cpu().numpy()
    nonnms_boxes = nonnms_boxes.cpu().numpy()
    nonnms_im_ids = nonnms_im_ids.cpu().numpy()

    Timer.get('Epoch', 'Batch', 'Detect', 'NMS').tic()
    box_inds, box_classes, box_class_scores, boxes = _box_results_with_nms_and_limit_np(nonnms_scores, nonnms_boxes, nonnms_im_ids)
    Timer.get('Epoch', 'Batch', 'Detect', 'NMS').toc()
    scores = nonnms_scores[box_inds, :]
    im_ids = nonnms_im_ids[box_inds].astype(np.int, copy=False)

    assert boxes.shape[0] > 0
    # assert np.all(np.stack([all_boxes[i, j*4:(j+1)*4] for i, j in zip(box_inds, classes)], axis=0) == boxes)

    Timer.get('Epoch', 'Batch', 'Detect', 'Mask').tic()
    masks = _im_detect_mask(model, im_scales_np, im_ids, boxes, feat_map)
    Timer.get('Epoch', 'Batch', 'Detect', 'Mask').toc()

    assert box_class_scores.shape[0] == boxes.shape[0] == box_classes.shape[0] == im_ids.shape[0] == masks.shape[0] == scores.shape[0]
    return box_class_scores, boxes, box_classes, im_ids, masks, feat_map, scores


def _im_detect_bbox(model, inputs):
    """Prepare the bbox for testing"""

    assert not cfg.PYTORCH_VERSION_LESS_THAN_040
    assert cfg.MODEL.FASTER_RCNN
    assert cfg.TEST.BBOX_REG
    assert not cfg.MODEL.CLS_AGNOSTIC_BBOX_REG
    assert not cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED

    Timer.get('Epoch', 'Batch', 'Detect', 'ImDetBox', 'Forward').tic()
    return_dict = model(**inputs)
    Timer.get('Epoch', 'Batch', 'Detect', 'ImDetBox', 'Forward').toc()
    Timer.get('Epoch', 'Batch', 'Detect', 'ImDetBox', 'Sync').tic()
    torch.cuda.synchronize()
    Timer.get('Epoch', 'Batch', 'Detect', 'ImDetBox', 'Sync').toc()
    box_deltas = return_dict['bbox_pred']
    scores = return_dict['cls_score']  # cls prob (activations after softmax)

    # try:  # Torch
    boxes = return_dict['rois'][:, 1:5]
    im_inds = return_dict['rois'][:, 0]
    im_scales = return_dict['roi_scales']
    # except AttributeError:  # Numpy
    #     boxes = box_deltas.new_tensor(return_dict['rois'][:, 1:5])
    #     im_inds = return_dict['rois'][:, 0].astype(np.int, copy=False)

    boxes /= im_scales.view(-1, 1)  # unscale back to raw image space
    scores = scores.view([-1, scores.shape[-1]])  # In case there is 1 proposal
    box_deltas = box_deltas.view([-1, box_deltas.shape[-1]])  # In case there is 1 proposal

    # Apply bounding-box regression deltas
    pred_boxes = bbox_transform(boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS, cfg.BBOX_XFORM_CLIP)
    pred_boxes = clip_tiled_boxes(pred_boxes, inputs['data'][0].shape[1:])

    return scores, pred_boxes, return_dict['blob_conv'], im_inds # NOTE: pred_boxes are scaled back to the original image size


def _box_results_with_nms_and_limit(all_scores, all_boxes, im_ids):
    """
    Returns bounding-box detection results by thresholding on scores and applying non-maximum suppression (NMS). In the following B denotes the
    number of detections and C the number of classes
    :param all_scores [array, B x C]: Each row represents a list of object detection confidence scores for each of the object classes in the dataset
                (including the background class). Element (i, j) corresponds to the box at `all_boxes[i, j * 4:(j + 1) * 4]`.
    :param all_boxes [array, B x 4C]: Each row represents a list of predicted bounding boxes for each of the object classes in  the dataset
                (including the background class). The detections in each row originate from the same object proposal.
    :param im_inds [array, R]: Which image the corresponding box belongs to.
    :return:
    """
    assert not cfg.TEST.SOFT_NMS.ENABLED
    assert not cfg.TEST.BBOX_VOTE.ENABLED

    num_classes = cfg.MODEL.NUM_CLASSES

    image_masks = [im_ids == im_id for im_id in torch.unique(im_ids, sorted=True)]
    all_boxes_ids = torch.arange(all_boxes.shape[0], dtype=torch.float32, device=all_boxes.device)

    # Apply threshold on detection probabilities and apply NMS. Skip j = 0, because it's the background class
    all_results = []
    for i, mask_i in enumerate(image_masks):
        scores_i = all_scores[mask_i, :]
        boxes_i = all_boxes[mask_i, :]
        boxes_ids_i = all_boxes_ids[mask_i]

        boxes_and_infos_per_class = {}
        for j in range(1, num_classes):
            class_boxes_mask = scores_i[:, j] > cfg.TEST.SCORE_THRESH

            scores_ij = scores_i[class_boxes_mask, j]
            if scores_ij.shape[0] > 0:
                boxes_ij = boxes_i[class_boxes_mask, j * 4:(j + 1) * 4]
                boxes_ids_ij = boxes_ids_i[class_boxes_mask]

                rois_ij = torch.cat([boxes_ij, scores_ij.view(-1, 1)], dim=1).float()
                keep = nms_gpu(rois_ij, cfg.TEST.NMS).long().squeeze(dim=1)
                # print(i, j, len(keep))

                scores_ij = scores_ij[keep]
                boxes_ids_ij = boxes_ids_ij[keep]
                boxes_ij = boxes_ij[keep, :]
                j_vec = torch.full_like(boxes_ids_ij, fill_value=j)
                box_infos_ij = torch.stack([scores_ij, boxes_ids_ij, j_vec], dim=1)
                boxes_and_infos_per_class[j] = torch.cat([boxes_ij, box_infos_ij], dim=1)

        # Limit to max_per_image detections **over all classes**
        if cfg.TEST.DETECTIONS_PER_IM > 0:
            image_scores = []
            for j in range(1, num_classes):
                if j in boxes_and_infos_per_class:
                    image_scores.append(boxes_and_infos_per_class[j][:, 4])
            image_scores = torch.cat(image_scores)
            if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
                image_thresh = torch.sort(image_scores)[0][-cfg.TEST.DETECTIONS_PER_IM]
                for j in range(1, num_classes):
                    if j in boxes_and_infos_per_class:
                        keep = boxes_and_infos_per_class[j][:, 4] >= image_thresh
                        boxes_and_infos_per_class[j] = boxes_and_infos_per_class[j][keep, :]

        im_result = torch.cat([boxes_and_infos_per_class[j] for j in range(1, num_classes) if j in boxes_and_infos_per_class], dim=0)
        all_results.append(im_result)

    all_results = torch.cat(all_results, dim=0)
    boxes_ids = all_results[:, 5].long()
    box_classes = all_results[:, 6].long()
    scores = all_results[:, 4]
    boxes = all_results[:, :4]
    return boxes_ids, box_classes, scores, boxes


def _box_results_with_nms_and_limit_np(all_scores, all_boxes, im_ids):
    """
    Returns bounding-box detection results by thresholding on scores and applying non-maximum suppression (NMS). In the following B denotes the
    number of detections and C the number of classes
    :param all_scores [array, B x C]: Each row represents a list of object detection confidence scores for each of the object classes in the dataset
                (including the background class). Element (i, j) corresponds to the box at `all_boxes[i, j * 4:(j + 1) * 4]`.
    :param all_boxes [array, B x 4C]: Each row represents a list of predicted bounding boxes for each of the object classes in  the dataset
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
    for i, mask_i in enumerate(image_masks):
        scores_i = all_scores[mask_i, :]
        boxes_i = all_boxes[mask_i, :]
        boxes_ids_i = all_boxes_ids[mask_i]

        boxes_and_infos_per_class = {}
        for j in range(1, num_classes):
            class_boxes_mask = scores_i[:, j] > cfg.TEST.SCORE_THRESH
            scores_ij = scores_i[class_boxes_mask, j]
            boxes_ij = boxes_i[class_boxes_mask, j * 4:(j + 1) * 4]
            boxes_ids_ij = boxes_ids_i[class_boxes_mask]

            rois_ij = np.concatenate([boxes_ij, scores_ij[:, None]], axis=1).astype(np.float32, copy=False)
            keep = box_utils.nms(rois_ij, cfg.TEST.NMS)
            # if np.any(class_boxes_mask):
            #     print(i, j, len(keep))

            scores_ij = scores_ij[keep]
            boxes_ids_ij = boxes_ids_ij[keep]
            boxes_ij = boxes_ij[keep, :]
            j_vec = np.full_like(boxes_ids_ij, fill_value=j)
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
        boxes (ndarray): B x 4 array of bounding box detections (e.g., as returned by im_detect_bbox)
        blob_conv (Variable): base features from the backbone network.

    Returns:
        pred_masks (ndarray): B x K x M x M array of class specific soft masks output by the network (must be processed by segm_results to convert
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
