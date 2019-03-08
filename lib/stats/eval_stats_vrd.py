from typing import List, Dict

import numpy as np

from lib.bbox_utils import compute_ious
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import Example
from lib.models.utils import Prediction


class EvalStatsVRD:
    class Counts:
        def __init__(self, num_images, num_predicates, num_thresholds):
            # The reason why I need to count GT matches and prediction hits separately is that a GT entry can be matched by more than one prediction
            # and one prediction can match more than one GT entry. Note that, for evaluation purposes, a prediction can only hit once,
            # even if multiple GT entries match.
            # !!! In the HICO-DET source code a "second hit" on a GT element is considered a false positive. FIXME modify?
            self.num_gt_matches = np.zeros([num_images, num_predicates, num_thresholds + 1])
            self.num_gt = np.zeros_like(self.num_gt_matches[:, :, 0])
            self.num_prediction_hits = np.zeros_like(self.num_gt_matches)
            self.num_predictions = np.zeros_like(self.num_gt_matches)

    def __init__(self, dataset: HicoDetInstanceSplit, triplet_matching_modes, iou_thresh=0.5):
        self.iou_thresh = iou_thresh
        self.triplet_matching_modes = triplet_matching_modes
        self.dataset = dataset
        self.nums_top_predictions = [20, 50, 100]

        self.counts = {('hoi-%s' % matching_mode): EvalStats.Counts(dataset.num_images, dataset.num_predicates, len(self.nums_top_predictions))
                       for matching_mode in self.triplet_matching_modes.keys()}
        self.counts['obj'] = EvalStats.Counts(dataset.num_images, dataset.num_object_classes, len(self.nums_top_predictions))

    @property
    def num_hoi_predictions_per_class(self):
        num_hoi_predictions_per_class__list = []
        for k in self.triplet_matching_modes.keys():
            num_hoi_predictions_per_class__list.append(np.sum(self.counts['hoi-%s' % k].num_predictions[:, :, -1], axis=0))
        num_hoi_predictions_per_class = num_hoi_predictions_per_class__list[0]

        # Note: this is across modes, not thresholds.
        assert all([np.all(num_hoi_predictions_per_class == num) for num in num_hoi_predictions_per_class__list])

        return num_hoi_predictions_per_class

    @property
    def num_obj_predictions_per_class(self):
        return np.sum(self.counts['obj'].num_predictions[:, :, -1], axis=0)

    @classmethod
    def evaluate_predictions(cls, dataset: HicoDetInstanceSplit, predictions: List[Dict], **kwargs):
        assert len(predictions) == dataset.num_images, (len(predictions), dataset.num_images)
        triplet_matching_modes = {'HOI': [0, 1, 2],
                                  'HI': [0, 1],
                                  'I': [1],
                                  }

        eval_stats = cls(dataset, triplet_matching_modes, **kwargs)
        for i, res in enumerate(predictions):
            ex = dataset.get_entry(i, read_img=False)
            prediction = Prediction.from_dict(res)
            eval_stats.process_prediction(i, ex, prediction)

        for count in eval_stats.counts.values():
            assert np.all(count.num_prediction_hits <= count.num_predictions)
            assert np.all(count.num_gt_matches <= count.num_gt[:, :, None])
            assert np.all(np.sum(count.num_gt, axis=0)) and np.all(np.sum(count.num_gt, axis=1))

        return eval_stats  # type: EvalStats

    def print(self):
        def _f(_x, _p):
            if _x < 1:
                if _x > 0:
                    return ('%{}.{}f%%'.format(_p + 3, _p)) % (_x * 100)
                else:
                    return ('%{}.{}f%%'.format(_p + 3, 0)) % (_x * 100)
            else:
                return ('%{}d%%'.format(_p + 3)) % 100

        for metric, counts in self.counts.items():
            print('{0} {1} (mode: {2}) {0}'.format('=' * 30, 'Evaluation results', metric))
            for k, num_top in enumerate(self.nums_top_predictions + [0]):
                print(('Top %d:' % num_top) if num_top > 0 else 'All:')

                # Global
                num_gt_matches_per_image = np.sum(counts.num_gt_matches[:, :, k], axis=1)
                num_gt_per_image = np.sum(counts.num_gt, axis=1)
                num_hits_per_image = np.sum(counts.num_prediction_hits[:, :, k], axis=1)
                num_predictions_per_image = np.sum(counts.num_predictions[:, :, k], axis=1)
                ap_per_image = np.divide(num_hits_per_image, num_predictions_per_image,
                                         out=np.zeros_like(num_hits_per_image),
                                         where=num_predictions_per_image > 0)
                print('    %10s: %s' % ('mAR', _f(np.mean(num_gt_matches_per_image / num_gt_per_image), _p=3)))
                print('    %10s: %s' % ('mAP', _f(np.mean(ap_per_image), _p=3)))

                # Per class
                num_gt_matches_per_class = np.sum(counts.num_gt_matches[:, :, k], axis=0)
                num_gt_per_class = np.sum(counts.num_gt, axis=0)
                num_hits_per_class = np.sum(counts.num_prediction_hits[:, :, k], axis=0)
                num_predictions_per_class = np.sum(counts.num_predictions[:, :, k], axis=0)
                ap_per_class = np.divide(num_hits_per_class, num_predictions_per_class,
                                         out=np.zeros_like(num_hits_per_class),
                                         where=num_predictions_per_class > 0)
                print('    %10s: [%s]' % ('pcAR', ' '.join([_f(arpc, _p=2) for arpc in (num_gt_matches_per_class / num_gt_per_class)])))
                print('    %10s: [%s]' % ('pcAP', ' '.join([_f(appc, _p=2) for appc in ap_per_class])))

    def process_prediction(self, img_idx, gt_entry: Example, prediction: Prediction):

        # After matching, values are sorted by score. Apply `inds` to the corresponding class field of prediction.

        for matching_mode, triplet_class_inds in self.triplet_matching_modes.items():
            gt_inds_per_prediction, inds = self.match_hois(gt_entry, prediction, triplet_class_inds)
            self._process_classes(img_idx, gt_inds_per_prediction, inds,
                                  gt_classes=gt_entry.gt_hois[:, 1],
                                  predicted_classes=prediction.hoi_classes,
                                  num_classes=self.dataset.num_predicates,
                                  counts=self.counts['hoi-%s' % matching_mode],
                                  )
        gt_inds_per_prediction, inds, predicted_obj_classes = self.match_objects(gt_entry, prediction)
        self._process_classes(img_idx, gt_inds_per_prediction, inds,
                              gt_classes=gt_entry.gt_obj_classes,
                              predicted_classes=predicted_obj_classes,
                              num_classes=self.dataset.num_object_classes,
                              counts=self.counts['obj'],
                              )

    def _process_classes(self, img_idx, gt_inds_per_prediction, inds, gt_classes, predicted_classes, num_classes, counts):
        counts.num_gt[img_idx, :] = np.array([np.sum(gt_classes == i) for i in range(num_classes)])
        assert np.sum(counts.num_gt[img_idx, :]) == gt_classes.shape[0]

        if not gt_inds_per_prediction:
            return

        num_predictions = len(gt_inds_per_prediction)
        assert num_predictions == predicted_classes.shape[0]
        predicted_classes = predicted_classes[inds]

        for k, num_top_preds in enumerate(self.nums_top_predictions + [num_predictions]):
            gt_inds_per_class = {}
            for pred_idx, curr_prediction_gt_matches in enumerate(gt_inds_per_prediction[:num_top_preds]):
                predicted_class = predicted_classes[pred_idx]
                assert all([gt_classes[gt_ind] == predicted_class for gt_ind in curr_prediction_gt_matches])
                gt_inds_per_class.setdefault(predicted_class, set()).update(curr_prediction_gt_matches)
                counts.num_prediction_hits[img_idx, predicted_class, k] += 1 if curr_prediction_gt_matches else 0
                counts.num_predictions[img_idx, predicted_class, k] += 1

            counts.num_gt_matches[img_idx, :, k] = np.array([len(gt_inds_per_class.get(hoi, [])) for hoi in range(num_classes)])
            assert np.all(counts.num_gt_matches[img_idx, :, k] <= counts.num_gt[img_idx, :])
            assert np.sum(counts.num_predictions[img_idx, :, k]) == min(num_top_preds, num_predictions)

    def match_objects(self, gt_entry: Example, prediction: Prediction):
        # TODO docs

        if isinstance(gt_entry, Example):
            gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)
            gt_obj_classes = gt_entry.gt_obj_classes
        else:
            raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))
        assert gt_boxes.shape[0] > 0

        predict_boxes = prediction.obj_boxes
        predict_obj_classes = prediction.obj_classes
        predict_obj_scores = prediction.obj_scores.max(axis=1)
        if predict_boxes is None:
            assert predict_obj_classes is None and predict_obj_scores is None
            return None, None
        assert len(np.unique(prediction.obj_im_inds)) == 1

        inds = np.argsort(predict_obj_scores)[::-1]
        predict_boxes = predict_boxes[inds, :]
        sorted_predict_obj_classes = predict_obj_classes[inds]

        # Compute matches. It's most efficient to match once and evaluate later.
        overlap = (compute_ious(gt_boxes, predict_boxes) >= self.iou_thresh)

        gt_inds_per_prediction = []
        for i, predicted_class in enumerate(sorted_predict_obj_classes):
            class_match = (gt_obj_classes == predicted_class)
            gt_inds_per_prediction.append(np.flatnonzero(overlap[:, i] & class_match).tolist())

        return gt_inds_per_prediction, inds, predict_obj_classes

    def match_hois(self, gt_entry: Example, prediction: Prediction, triplet_class_inds):
        # TODO docs

        if isinstance(gt_entry, Example):
            gt_hois = gt_entry.gt_hois[:, [0, 2, 1]]  # (h, o, i)
            gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)
            gt_obj_classes = gt_entry.gt_obj_classes
        else:
            raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))

        if not prediction.is_complete():
            return None, None
        assert len(np.unique(prediction.obj_im_inds)) == len(np.unique(prediction.hoi_img_inds)) == 1

        predict_obj_classes = prediction.obj_classes
        predict_obj_scores = prediction.obj_scores.max(axis=1)
        predict_boxes = prediction.obj_boxes

        predict_ho_pairs = prediction.ho_pairs
        predict_hois = np.concatenate([predict_ho_pairs, prediction.hoi_classes[:, None]], axis=1)
        predict_hoi_scores = prediction.hoi_score_distributions.max(axis=1)

        assert gt_hois.shape[0] > 0
        gt_triplets, gt_triplet_boxes = self.to_triplet(gt_hois, gt_obj_classes, gt_boxes)
        pred_triplets, pred_triplet_boxes, scores, inds = self.to_triplet(predict_hois, predict_obj_classes, predict_boxes, predict_hoi_scores,
                                                                          predict_obj_scores)

        if not np.all(scores[1:] <= scores[:-1]):
            raise ValueError("Somehow the relations werent sorted properly")

        # Compute matches. It's most efficient to match once and evaluate later.
        gt_inds_per_prediction = self._match_hoi_triplets(gt_triplets, pred_triplets, gt_triplet_boxes, pred_triplet_boxes, triplet_class_inds)

        return gt_inds_per_prediction, inds

    def _match_hoi_triplets(self, gt_triplets, pred_triplets, gt_boxes, pred_boxes, triplet_class_inds):

        # The rows correspond to GT triplets, columns to pred triplets. Elements are booleans indicating a match.
        matches = (gt_triplets[:, triplet_class_inds, None] == pred_triplets.T[None, triplet_class_inds, :]).all(axis=1)

        gt_has_match = matches.any(axis=1)
        gt_inds_per_prediction = [[] for x in range(pred_boxes.shape[0])]
        for gt_ind, gt_box, keep_inds in zip(np.flatnonzero(gt_has_match),
                                             gt_boxes[gt_has_match],
                                             matches[gt_has_match]):
            boxes = pred_boxes[keep_inds]
            sub_ious = np.squeeze(compute_ious(gt_box[None, :4], boxes[:, :4]), axis=0)
            obj_ious = np.squeeze(compute_ious(gt_box[None, 4:], boxes[:, 4:]), axis=0)
            inds = (sub_ious >= self.iou_thresh) & (obj_ious >= self.iou_thresh)

            for i in np.flatnonzero(keep_inds)[inds]:
                gt_inds_per_prediction[i].append(int(gt_ind))
        return gt_inds_per_prediction

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
