from typing import List, Dict

import numpy as np

from config import cfg
from lib.bbox_utils import compute_ious
from lib.containers import Minibatch, Prediction, Example
from lib.dataset.hicodet import HicoDetInstance


class ResultStatistics:
    def __init__(self, dataset: HicoDetInstance, use_gt_boxes=None, iou_thresh=0.5):
        if use_gt_boxes is None:
            use_gt_boxes = cfg.program.predcls
        self.use_gt_boxes = use_gt_boxes
        self.iou_thresh = iou_thresh

        self.nums_top_predictions = [20, 50, 100]

        # The reason why I need to count GT matches and prediction hits separately is that a GT entry can be matched by more than one prediction and
        # one prediction can match more than one GT entry. Note that, for evaluation purposes, a prediction can only hit once, even if multiple GT
        # entries match.
        self.num_gt_matches = np.zeros([dataset.num_images, dataset.num_predicates, len(self.nums_top_predictions) + 1])
        self.num_gt = np.zeros_like(self.num_gt_matches[:, :, 0])
        self.num_prediction_hits = np.zeros_like(self.num_gt_matches)
        self.num_predictions = np.zeros_like(self.num_gt_matches)

    def _record_prediction(self, img_idx, gt_entry: Example, prediction: Prediction):
        predicted_hoi_to_gt = find_pred_to_gt_matches(gt_entry, prediction, use_gt_boxes=self.use_gt_boxes, iou_thresh=self.iou_thresh)
        self.num_gt[img_idx, :] = np.array([np.sum(gt_entry.gt_hois[:, 1] == i) for i in range(self.num_gt.shape[1])])
        assert np.sum(self.num_gt[img_idx, :]) == gt_entry.gt_hois.shape[0]

        if not predicted_hoi_to_gt:
            return

        num_predictions = len(predicted_hoi_to_gt)
        assert num_predictions == prediction.ho_pairs.shape[0]
        predicted_hoi_classes = prediction.hoi_classes

        for k, num_top_preds in enumerate(self.nums_top_predictions + [num_predictions]):
            matched_gt_inds_per_class = {}
            for pred_idx, curr_prediction_gt_matches in enumerate(predicted_hoi_to_gt[:num_top_preds]):
                pred_hoi_class = predicted_hoi_classes[pred_idx]
                assert all([gt_entry.gt_hois[gt_ind, 1] == pred_hoi_class for gt_ind in curr_prediction_gt_matches])
                matched_gt_inds_per_class.setdefault(pred_hoi_class, set()).update(curr_prediction_gt_matches)
                self.num_predictions[img_idx, pred_hoi_class, k] += 1
                self.num_prediction_hits[img_idx, pred_hoi_class, k] += 1 if curr_prediction_gt_matches else 0

            self.num_gt_matches[img_idx, :, k] = np.array([len(matched_gt_inds_per_class.get(hoi, [])) for hoi in range(self.num_gt.shape[1])])
            assert np.all(self.num_gt_matches[img_idx, :, k] <= self.num_gt[img_idx, :])

    # noinspection PyStringFormat
    def print(self):
        print('{0} {1} {0}'.format('=' * 30, 'Evaluation results'))
        for k, num_top in enumerate(self.nums_top_predictions + [0]):
            # Global
            num_gt_matches_per_image = np.sum(self.num_gt_matches[:, :, k], axis=1)
            num_gt_per_image = np.sum(self.num_gt, axis=1)
            num_hits_per_image = np.sum(self.num_prediction_hits[:, :, k], axis=1)
            num_predictions_per_image = np.sum(self.num_predictions[:, :, k], axis=1)
            ap_per_image = np.divide(num_hits_per_image, num_predictions_per_image,
                                     out=np.zeros_like(num_hits_per_image),
                                     where=num_predictions_per_image > 0)
            print(('Top %d:' % num_top) if num_top > 0 else 'All:')
            print('    %10s: %.3f%%' % ('mAR', 100 * np.mean(num_gt_matches_per_image / num_gt_per_image)))
            print('    %10s: %.3f%%' % ('mAP', 100 * np.mean(ap_per_image)))

    @classmethod
    def evaluate_predictions(cls, dataset: HicoDetInstance, predictions: List[Dict], **kwargs):
        assert len(predictions) == dataset.num_images, (len(predictions), dataset.num_images)
        evaluator = cls(dataset, **kwargs)
        for i, res in enumerate(predictions):
            ex = dataset.get_entry(i, read_img=False)
            prediction = Prediction(**res)
            evaluator._record_prediction(i, ex, prediction)
            if i % 100 == 0:
                print(i)
        assert np.all(evaluator.num_prediction_hits <= evaluator.num_predictions)
        assert np.all(evaluator.num_gt_matches <= evaluator.num_gt[:, :, None])
        assert np.all(np.sum(evaluator.num_gt, axis=0)) and np.all(np.sum(evaluator.num_gt, axis=1))
        return evaluator


def find_pred_to_gt_matches(gt_entry: Example, prediction: Prediction, use_gt_boxes=False, **kwargs):
    # TODO docs

    if isinstance(gt_entry, Minibatch):
        assert False
        # im_scales = gt_entry.img_infos[:, 2].cpu().numpy()
        # gt_hois = gt_entry.gt_hois[:, [0, 2, 1]]  # (h, o, i)
        # gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False) / im_scales[gt_entry.gt_box_im_ids, None]
        # gt_obj_classes = gt_entry.gt_obj_classes
    elif isinstance(gt_entry, Example):
        gt_hois = gt_entry.gt_hois[:, [0, 2, 1]]  # (h, o, i)
        gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)
        gt_obj_classes = gt_entry.gt_obj_classes
    else:
        raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))

    if not prediction.is_complete():
        return []
    assert len(np.unique(prediction.obj_im_inds)) == len(np.unique(prediction.hoi_img_inds)) == 1

    if use_gt_boxes:
        predict_boxes = gt_boxes
        predict_obj_classes = gt_obj_classes
        predict_obj_scores = np.ones(predict_obj_classes.shape[0])
    else:
        predict_boxes = prediction.obj_boxes
        predict_obj_score_dists = prediction.obj_scores
        predict_obj_classes = predict_obj_score_dists.argmax(axis=1)
        predict_obj_scores = predict_obj_score_dists.max(axis=1)

    predict_ho_pairs = prediction.ho_pairs
    predict_hois = np.concatenate([predict_ho_pairs, prediction.hoi_classes[:, None]], axis=1)
    predict_hoi_scores = prediction.hoi_score_distributions.max(axis=1)

    pred_to_gt = find_pred_to_gt_match_on_triplets(gt_hois, gt_boxes, gt_obj_classes,
                                                   predict_hois, predict_boxes, predict_obj_classes, predict_hoi_scores, predict_obj_scores,
                                                   **kwargs)
    return pred_to_gt


def find_pred_to_gt_match_on_triplets(gt_hois, gt_boxes, gt_obj_classes,
                                      predict_hois, predict_boxes, predict_box_classes, hoi_scores, predict_box_scores,
                                      iou_thresh=0.5):
    num_gt_relations = gt_hois.shape[0]
    assert num_gt_relations > 0

    gt_triplets, gt_triplet_boxes = to_triples(gt_hois, gt_obj_classes, gt_boxes)
    pred_triplets, pred_triplet_boxes, scores_overall = to_triples(predict_hois, predict_box_classes, predict_boxes, hoi_scores, predict_box_scores)

    if not np.all(scores_overall[1:] <= scores_overall[:-1]):
        raise ValueError("Somehow the relations werent sorted properly")

    # Compute recall. It's most efficient to match once and then do recall after
    pred_to_gt = _compute_pred_matches(gt_triplets, pred_triplets, gt_triplet_boxes, pred_triplet_boxes, iou_thresh)

    return pred_to_gt


def to_triples(hois, obj_classes, boxes, hoi_scores=None, obj_scores=None):
    actions = hois[:, 2]
    ho_pairs = hois[:, :2]

    ho_classes = obj_classes[ho_pairs]
    hois = np.stack((ho_classes[:, 0], actions, ho_classes[:, 1]), axis=1)
    ho_boxes = np.concatenate((boxes[ho_pairs[:, 0]], boxes[ho_pairs[:, 1]]), axis=1)

    if hoi_scores is not None and obj_scores is not None:
        hoi_scores *= obj_scores[ho_pairs[:, 0]] * obj_scores[ho_pairs[:, 1]]
        inds = np.argsort(hoi_scores)[::-1]
        hois = hois[inds]
        ho_boxes = ho_boxes[inds]
        hoi_scores = hoi_scores[inds]
        return hois, ho_boxes, hoi_scores
    else:
        return hois, ho_boxes


def _compute_pred_matches(gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thresh):
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = (gt_triplets[..., None] == pred_triplets.T[None, ...]).all(axis=1)
    gt_has_match = keeps.any(axis=1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.flatnonzero(gt_has_match),
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match]):
        boxes = pred_boxes[keep_inds]
        sub_ious = np.squeeze(compute_ious(gt_box[None, :4], boxes[:, :4]), axis=0)
        obj_ious = np.squeeze(compute_ious(gt_box[None, 4:], boxes[:, 4:]), axis=0)
        inds = (sub_ious >= iou_thresh) & (obj_ious >= iou_thresh)

        for i in np.flatnonzero(keep_inds)[inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt
