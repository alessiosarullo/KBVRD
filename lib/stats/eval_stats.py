from typing import List, Dict

import numpy as np

from config import cfg
from lib.bbox_utils import compute_ious
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import Example, Minibatch
from lib.models.utils import Prediction


class EvalStats:
    triplet_matching_modes = {'HOI': [0, 1, 2],
                              'HI': [0, 1]}

    def __init__(self, dataset: HicoDetInstanceSplit, triplet_class_inds, iou_thresh=0.5):
        self.use_gt_boxes = cfg.program.predcls
        self.iou_thresh = iou_thresh
        self.triplet_class_inds = triplet_class_inds
        self.dataset = dataset
        self.nums_top_predictions = [20, 50, 100]

        # The reason why I need to count GT matches and prediction hits separately is that a GT entry can be matched by more than one prediction and
        # one prediction can match more than one GT entry. Note that, for evaluation purposes, a prediction can only hit once, even if multiple GT
        # entries match.
        self.num_gt_matches = np.zeros([dataset.num_images, dataset.num_predicates, len(self.nums_top_predictions) + 1])
        self.num_gt = np.zeros_like(self.num_gt_matches[:, :, 0])
        self.num_prediction_hits = np.zeros_like(self.num_gt_matches)
        self.num_predictions = np.zeros_like(self.num_gt_matches)

    def _record_prediction(self, img_idx, gt_entry: Example, prediction: Prediction):
        predicted_hoi_to_gt, inds = self.find_pred_to_gt_matches(gt_entry, prediction)
        # Predicates are sorted by score now. To match apply `inds` to every HOI field of prediction.

        self.num_gt[img_idx, :] = np.array([np.sum(gt_entry.gt_hois[:, 1] == i) for i in range(self.dataset.num_predicates)])
        assert np.sum(self.num_gt[img_idx, :]) == gt_entry.gt_hois.shape[0]

        if not predicted_hoi_to_gt:
            return

        num_predictions = len(predicted_hoi_to_gt)
        assert num_predictions == prediction.ho_pairs.shape[0]
        predicted_hoi_classes = prediction.hoi_classes[inds]

        for k, num_top_preds in enumerate(self.nums_top_predictions + [num_predictions]):
            matched_gt_inds_per_class = {}
            for pred_idx, curr_prediction_gt_matches in enumerate(predicted_hoi_to_gt[:num_top_preds]):
                pred_hoi_class = predicted_hoi_classes[pred_idx]
                assert all([gt_entry.gt_hois[gt_ind, 1] == pred_hoi_class for gt_ind in curr_prediction_gt_matches])
                matched_gt_inds_per_class.setdefault(pred_hoi_class, set()).update(curr_prediction_gt_matches)
                self.num_prediction_hits[img_idx, pred_hoi_class, k] += 1 if curr_prediction_gt_matches else 0
                self.num_predictions[img_idx, pred_hoi_class, k] += 1

            self.num_gt_matches[img_idx, :, k] = np.array([len(matched_gt_inds_per_class.get(hoi, [])) for hoi in range(self.dataset.num_predicates)])
            assert np.all(self.num_gt_matches[img_idx, :, k] <= self.num_gt[img_idx, :])
            assert np.sum(self.num_predictions[img_idx, :, k]) == min(num_top_preds, num_predictions)

    @classmethod
    def print(cls, eval_stats_dict):
        def _f(_x, _p):
            if _x < 1:
                if _x > 0:
                    return ('%{}.{}f%%'.format(_p + 3, _p)) % (_x * 100)
                else:
                    return ('%{}.{}f%%'.format(_p + 3, 0)) % (_x * 100)
            else:
                return '100%'

        for matching_mode, eval_stats in eval_stats_dict.items():
            print('{0} {1} (matching {2}) {0}'.format('=' * 30, 'Evaluation results', matching_mode))
            for k, num_top in enumerate(eval_stats.nums_top_predictions + [0]):
                print(('Top %d:' % num_top) if num_top > 0 else 'All:')

                # Global
                num_gt_matches_per_image = np.sum(eval_stats.num_gt_matches[:, :, k], axis=1)
                num_gt_per_image = np.sum(eval_stats.num_gt, axis=1)
                num_hits_per_image = np.sum(eval_stats.num_prediction_hits[:, :, k], axis=1)
                num_predictions_per_image = np.sum(eval_stats.num_predictions[:, :, k], axis=1)
                ap_per_image = np.divide(num_hits_per_image, num_predictions_per_image,
                                         out=np.zeros_like(num_hits_per_image),
                                         where=num_predictions_per_image > 0)
                print('    %10s: %s' % ('mAR', _f(np.mean(num_gt_matches_per_image / num_gt_per_image), _p=3)))
                print('    %10s: %s' % ('mAP', _f(np.mean(ap_per_image), _p=3)))

                # Per class
                num_gt_matches_per_class = np.sum(eval_stats.num_gt_matches[:, :, k], axis=0)
                num_gt_per_class = np.sum(eval_stats.num_gt, axis=0)
                num_hits_per_class = np.sum(eval_stats.num_prediction_hits[:, :, k], axis=0)
                num_predictions_per_class = np.sum(eval_stats.num_predictions[:, :, k], axis=0)
                ap_per_class = np.divide(num_hits_per_class, num_predictions_per_class,
                                         out=np.zeros_like(num_hits_per_class),
                                         where=num_predictions_per_class > 0)
                print('    %10s: [%s]' % ('pcAR', ' '.join([_f(arpc, _p=2) for arpc in (num_gt_matches_per_class / num_gt_per_class)])))
                print('    %10s: [%s]' % ('pcAP', ' '.join([_f(appc, _p=2) for appc in ap_per_class])))

    @classmethod
    def evaluate_predictions(cls, dataset: HicoDetInstanceSplit, predictions: List[Dict], **kwargs):
        assert len(predictions) == dataset.num_images, (len(predictions), dataset.num_images)
        all_eval_stats = {}
        for k, v in cls.triplet_matching_modes.items():
            eval_stats = cls(dataset, triplet_class_inds=v, **kwargs)
            for i, res in enumerate(predictions):
                ex = dataset.get_entry(i, read_img=False)
                prediction = Prediction.from_dict(res)
                eval_stats._record_prediction(i, ex, prediction)
            assert np.all(eval_stats.num_prediction_hits <= eval_stats.num_predictions)
            assert np.all(eval_stats.num_gt_matches <= eval_stats.num_gt[:, :, None])
            assert np.all(np.sum(eval_stats.num_gt, axis=0)) and np.all(np.sum(eval_stats.num_gt, axis=1))
            all_eval_stats[k] = eval_stats
        return all_eval_stats

    def find_pred_to_gt_matches(self, gt_entry: Example, prediction: Prediction):
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
            return None, None
        assert len(np.unique(prediction.obj_im_inds)) == len(np.unique(prediction.hoi_img_inds)) == 1

        if self.use_gt_boxes:
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

        assert gt_hois.shape[0] > 0
        gt_triplets, gt_triplet_boxes = self.to_triplet(gt_hois, gt_obj_classes, gt_boxes)
        pred_triplets, pred_triplet_boxes, scores, inds = self.to_triplet(predict_hois, predict_obj_classes, predict_boxes, predict_hoi_scores,
                                                                          predict_obj_scores)

        if not np.all(scores[1:] <= scores[:-1]):
            raise ValueError("Somehow the relations werent sorted properly")

        # Compute recall. It's most efficient to match once and then do recall after
        pred_to_gt = self._compute_pred_matches(gt_triplets, pred_triplets, gt_triplet_boxes, pred_triplet_boxes)

        return pred_to_gt, inds

    def _compute_pred_matches(self, gt_triplets, pred_triplets, gt_boxes, pred_boxes):

        # The rows correspond to GT triplets, columns to pred triplets. Elements are booleans indicating a match.
        matches = (gt_triplets[:, self.triplet_class_inds, None] == pred_triplets.T[None, self.triplet_class_inds, :]).all(axis=1)

        gt_has_match = matches.any(axis=1)
        pred_to_gt = [[] for x in range(pred_boxes.shape[0])]
        for gt_ind, gt_box, keep_inds in zip(np.flatnonzero(gt_has_match),
                                             gt_boxes[gt_has_match],
                                             matches[gt_has_match]):
            boxes = pred_boxes[keep_inds]
            sub_ious = np.squeeze(compute_ious(gt_box[None, :4], boxes[:, :4]), axis=0)
            obj_ious = np.squeeze(compute_ious(gt_box[None, 4:], boxes[:, 4:]), axis=0)
            inds = (sub_ious >= self.iou_thresh) & (obj_ious >= self.iou_thresh)

            for i in np.flatnonzero(keep_inds)[inds]:
                pred_to_gt[i].append(int(gt_ind))
        return pred_to_gt

    @staticmethod
    def to_triplet(hois, obj_classes, boxes, hoi_scores=None, obj_scores=None):
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
            return hois, ho_boxes, hoi_scores, inds
        else:
            return hois, ho_boxes
