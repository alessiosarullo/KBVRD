"""
Adapted from Danfei Xu. In particular, slow code was removed
"""
import numpy as np
from lib.containers import Minibatch, Prediction
from lib.bbox_utils import compute_ious
import torch


# TODO rename

class Evaluator:
    def __init__(self):
        self.result_dict = {'recall': {20: [], 50: [], 100: []}}
        np.set_printoptions(precision=3)

    def evaluate_scene_graph_entry(self, example, pred_scores, iou_thresh=0.5):
        pred_to_gt = self.evaluate_from_dict(example, pred_scores, iou_thresh=iou_thresh)
        for k in self.result_dict['recall']:
            matched_gt_inds = set([gt_ind for p2g in pred_to_gt[:k] for gt_ind in p2g])
            recall_i = len(matched_gt_inds) / example.gt_hois.shape[0]
            self.result_dict['recall'][k].append(recall_i)

    def print_stats(self):
        print('{0} {1} {0}'.format('=' * 30, 'Evaluation results'))
        for k, v in self.result_dict['recall'].items():
            print('R@%i: %f' % (k, np.mean(v)))

    @staticmethod
    def evaluate_from_dict(example: Minibatch, prediction: Prediction, **kwargs):
        gt_hois = example.gt_hois
        gt_boxes = example.gt_boxes.astype(np.float, copy=False)
        gt_obj_classes = example.gt_obj_classes

        predict_ho_pairs = prediction.ho_pairs
        predict_hoi_scores = prediction.hoi_scores.cpu().numpy()

        predict_boxes = prediction.obj_boxes
        predict_obj_scores = prediction.obj_scores
        predict_obj_classes = predict_obj_scores.argmax(dim=1).cpu().numpy()

        predict_hoi = np.concatenate([predict_ho_pairs, predict_hoi_scores.argmax(axis=1, keepdim=True)], axis=1)
        predict_hoi_score = predict_hoi_scores.max(axis=1)

        pred_to_gt = evaluate_recall(gt_hois, gt_boxes, gt_obj_classes,
                                     predict_hoi, predict_boxes, predict_obj_classes, predict_hoi_score, predict_obj_scores,
                                     **kwargs)

        return pred_to_gt


def evaluate_recall(gt_hois, gt_boxes, gt_obj_classes,
                    predict_hois, predict_boxes, predict_box_classes, hoi_scores, predict_box_scores,
                    iou_thresh=0.5):
    if predict_hois.size == 0:
        return [[]]

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
    ho_boxes = np.stack((boxes[ho_pairs[:, 0]], boxes[ho_pairs[:, 1]]), axis=1)

    if hoi_scores is not None and obj_scores is not None:
        hoi_scores *= obj_scores[ho_pairs[:, 0]] * obj_scores[ho_pairs[:, 1]]
        return hois, ho_boxes, hoi_scores
    else:
        return hois, ho_boxes


def _compute_pred_matches(gt_triplets, pred_triplets, gt_boxes, pred_boxes, iou_thresh):
    # This performs a matrix multiplication-esque thing between the two arrays
    # Instead of summing, we want the equality, so we reduce in that way
    # The rows correspond to GT triplets, columns to pred triplets
    keeps = (gt_triplets[..., None] == pred_triplets.T[None, ...]).all(axis=1)
    gt_has_match = keeps.any(dim=1)
    pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
    for gt_ind, gt_box, keep_inds in zip(np.flatnonzero(gt_has_match),
                                         gt_boxes[gt_has_match],
                                         keeps[gt_has_match]):
        boxes = pred_boxes[keep_inds]
        sub_ious = compute_ious(gt_box[None, :4], boxes[:, :4])
        obj_ious = compute_ious(gt_box[None, 4:], boxes[:, 4:])
        inds = (sub_ious >= iou_thresh) & (obj_ious >= iou_thresh)

        for i in np.flatnonzero(keep_inds)[inds]:
            pred_to_gt[i].append(int(gt_ind))
    return pred_to_gt
