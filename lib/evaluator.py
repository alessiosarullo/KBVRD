import numpy as np

from config import cfg
from lib.bbox_utils import compute_ious
from lib.containers import Minibatch, Prediction, Example
from lib.dataset.hicodet import HicoDetInstance

from typing import List, Dict, Union


# TODO rename
class Evaluator:
    def __init__(self, use_gt_boxes=None, iou_thresh=0.5):
        if use_gt_boxes is None:
            use_gt_boxes = cfg.program.predcls
        self.result_dict = {'recall': {20: [], 50: [], 100: []}}
        self.use_gt_boxes = use_gt_boxes
        self.iou_thresh = iou_thresh

    def evaluate_prediction(self, example, prediction):
        predicted_hoi_to_gt = find_pred_to_gt_matches(example, prediction, use_gt_boxes=self.use_gt_boxes, iou_thresh=self.iou_thresh)
        for k in self.result_dict['recall']:
            matched_gt_inds = set([gt_ind for p2g in predicted_hoi_to_gt[:k] for gt_ind in p2g])
            recall_i = len(matched_gt_inds) / example.gt_hois.shape[0]
            self.result_dict['recall'][k].append(recall_i)

    def print_stats(self):
        print('{0} {1} {0}'.format('=' * 30, 'Evaluation results'))
        for k, v in self.result_dict['recall'].items():
            print('R@%3d: %.3f%%' % (k, 100 * np.mean(v)))

    @classmethod
    def evaluate_predictions(cls, dataset: HicoDetInstance, predictions: List[Dict], **kwargs):
        assert len(predictions) == len(dataset), (len(predictions), len(dataset))
        evaluator = cls(**kwargs)
        for i, res in enumerate(predictions):
            ex = dataset.get_entry(i, read_img=False)
            prediction = Prediction(**res)
            evaluator.evaluate_prediction(ex, prediction)
            if i % 100 == 0:
                print(i)
        return evaluator


def find_pred_to_gt_matches(gt_entry: Union[Minibatch, Example], prediction: Prediction, use_gt_boxes=False, **kwargs):
    # TODO docs

    if not prediction.is_complete():
        return [[]]
    assert len(np.unique(prediction.obj_im_inds)) == len(np.unique(prediction.hoi_img_inds)) == 1

    if isinstance(gt_entry, (Minibatch, Example)):
        gt_hois = gt_entry.gt_hois[:, [0, 2, 1]]
        gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)
        gt_obj_classes = gt_entry.gt_obj_classes
    elif isinstance(gt_entry, Example):
        gt_hois = gt_entry.gt_hois[:, [0, 2, 1]]
        gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)
        gt_obj_classes = gt_entry.gt_obj_classes
    else:
        raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))

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
    predict_hoi_score_dists = prediction.hoi_scores
    predict_hois = np.concatenate([predict_ho_pairs, predict_hoi_score_dists.argmax(axis=1)[:, None]], axis=1)
    predict_hoi_scores = predict_hoi_score_dists.max(axis=1)

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
