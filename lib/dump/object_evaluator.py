from collections import Counter
from typing import List, Dict

import numpy as np

from lib.bbox_utils import compute_ious
from lib.dataset.hicodet.hicodet_split import HicoDetSplitBuilder, HicoDetSplit, Example
from lib.models.containers import Prediction
from lib.stats.evaluator import BaseEvaluator


class ObjectEvaluator(BaseEvaluator):
    def __init__(self, dataset: HicoDetSplit, iou_thresh=0.5, hoi_score_thr=None, num_hoi_thr=None):
        super().__init__()
        self.iou_thresh = iou_thresh
        self.hoi_score_thr = hoi_score_thr
        self.num_hoi_thr = num_hoi_thr

        self.dataset = dataset

        self.gt_classes = []
        self.predict_scores = []
        self.pred_gt_assignment = []
        self.gt_count = 0

        self.metrics = {}  # type: Dict[str, np.ndarray]

    def evaluate_predictions(self, predictions: List[Dict], **kwargs):
        assert len(predictions) == self.dataset.num_images, (len(predictions), self.dataset.num_images)

        for i, res in enumerate(predictions):
            ex = self.dataset.get_img_entry(i, read_img=False)
            prediction = Prediction(res)
            self.process_prediction(i, ex, prediction)
        self.compute_metrics()

    def compute_metrics(self):
        gt_classes = np.concatenate(self.gt_classes, axis=0)
        assert self.gt_count == gt_classes.shape[0]
        predict_scores = np.concatenate(self.predict_scores, axis=0)
        pred_gt_assignment = np.concatenate(self.pred_gt_assignment, axis=0)

        gt_hoi_classes_count = Counter(gt_classes.tolist())

        ap = np.zeros(self.dataset.num_object_classes)
        recall = np.zeros(self.dataset.num_object_classes)
        for j in range(self.dataset.num_object_classes):
            scores = predict_scores[:, j]
            assignment = pred_gt_assignment[:, j]
            if self.hoi_score_thr is not None:
                inds = (scores >= self.hoi_score_thr)
                scores = scores[inds]
                assignment = assignment[inds]
            if self.num_hoi_thr is not None:
                scores = scores[:self.num_hoi_thr]
                assignment = assignment[:self.num_hoi_thr]
            rec_j, prec_j, ap_j = self.eval_interactions(scores, assignment, gt_hoi_classes_count[j])
            ap[j] = ap_j
            if rec_j.size > 0:
                recall[j] = rec_j[-1]

        self.metrics['obj-M-mAP'] = ap
        self.metrics['obj-M-rec'] = recall

    def print_metrics(self, sort=False):
        mf = MetricFormatter()
        lines = []

        obj_metrics = {k: v for k, v in self.metrics.items() if k.lower().startswith('obj')}
        lines += mf.format_metric_and_gt_lines(self.dataset.obj_labels, metrics=obj_metrics, gt_str='GT objects', sort=sort)

        printstr = '\n'.join(lines)
        print(printstr)
        return printstr

    def process_prediction(self, im_id, gt_entry: Example, prediction: Prediction):
        if isinstance(gt_entry, Example):
            gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)
            gt_classes = gt_entry.gt_obj_classes
            num_gt = gt_boxes.shape[0]
            gt_ho_ids = self.gt_count + np.arange(num_gt)
            self.gt_count += num_gt
        else:
            raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))
        assert gt_classes is not None

        predict_scores = np.zeros([0, self.dataset.num_object_classes])
        predict_boxes = np.zeros((0, 4))
        if prediction.obj_boxes is not None:
            assert prediction.obj_im_inds.shape[0] == prediction.obj_boxes.shape[0]
            predict_boxes = prediction.obj_boxes
            predict_scores = prediction.obj_scores

        pred_gt_ious = compute_ious(predict_boxes, gt_boxes)
        pred_gt_assignment = np.full((predict_scores.shape[0], self.dataset.num_object_classes), fill_value=-1, dtype=np.int)
        for predict_idx in range(predict_scores.shape[0]):
            gt_ious = pred_gt_ious[predict_idx]
            if np.any(gt_ious >= self.iou_thresh):
                gt_ious_per_class = np.zeros((num_gt, self.dataset.num_object_classes))
                gt_ious_per_class[np.arange(num_gt), gt_classes] = gt_ious
                gt_assignments = gt_ious_per_class.argmax(axis=0)[np.any(gt_ious_per_class >= self.iou_thresh, axis=0)]
                gt_class_assignments = gt_classes[gt_assignments]
                assert np.unique(gt_assignments).size == gt_assignments.size
                assert np.unique(gt_class_assignments).size == gt_class_assignments.size
                pred_gt_assignment[predict_idx, gt_class_assignments] = gt_ho_ids[gt_assignments]

        self.gt_classes.append(gt_classes)
        self.predict_scores.append(predict_scores)
        self.pred_gt_assignment.append(pred_gt_assignment)

    def eval_interactions(self, predicted_conf_scores, pred_gtid_assignment, num_hoi_gt_positives):
        num_predictions = predicted_conf_scores.shape[0]
        tp = np.zeros(num_predictions)

        if num_predictions > 0:
            inds = np.argsort(predicted_conf_scores)[::-1]
            pred_gtid_assignment = pred_gtid_assignment[inds]

            matched_gt_inds, highest_scoring_pred_idx_per_gt_ind = np.unique(pred_gtid_assignment, return_index=True)
            if matched_gt_inds[0] == -1:
                matched_gt_inds = matched_gt_inds[1:]
                highest_scoring_pred_idx_per_gt_ind = highest_scoring_pred_idx_per_gt_ind[1:]
            tp[highest_scoring_pred_idx_per_gt_ind] = 1

        fp = 1 - tp
        assert np.all(fp[pred_gtid_assignment < 0] == 1)

        # compute precision/recall
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp / num_hoi_gt_positives
        prec = tp / (fp + tp)

        # compute average precision
        ap = 0
        for t in np.arange(11) / 10:
            pr = prec[rec >= t]
            if pr.size > 0:
                p = max(pr)
            else:
                p = 0
            ap = ap + p / 11
        return rec, prec, ap


class MetricFormatter:
    def __init__(self):
        super().__init__()

    def format_metric_and_gt_lines(self, gt_labels, metrics, gt_str, sort=False):
        lines = []
        gt_label_hist = Counter(gt_labels)
        num_gt_examples = sum(gt_label_hist.values())
        if sort:
            inds = [p for p, num in gt_label_hist.most_common()]
        else:
            inds = range(len(gt_label_hist))

        for k, v in metrics.items():
            assert v.size == 1 or len(inds) == v.size
            lines += [self.format_metric(k, v[inds] if v.size > 1 else v, len(gt_str))]
        format_str = '%{}s %8s [%s]'.format(len(gt_str) + 1)
        lines += [format_str % ('%s:' % gt_str, 'IDs', ' '.join(['%5d ' % i for i in inds]))]
        lines += [format_str % ('', '%', ' '.join([self._format_percentage(gt_label_hist[i] / num_gt_examples) for i in inds]))]
        return lines

    def format_metric(self, metric_name, data, metric_str_len=None):
        metric_str_len = metric_str_len or len(metric_name)
        per_class_str = ' @ [%s]' % ' '.join([self._format_percentage(x) for x in data]) if data.size > 1 else ''
        f_str = ('%{}s: %s%s'.format(metric_str_len)) % (metric_name, self._format_percentage(np.mean(data)), per_class_str)
        return f_str

    @staticmethod
    def _format_percentage(value, precision=2):
        if value < 1:
            if value > 0:
                return ('%{}.{}f%%'.format(precision + 3, precision)) % (value * 100)
            else:
                return ('%{}.{}f%%'.format(precision + 3, 0)) % (value * 100)
        else:
            return ('%{}d%%'.format(precision + 3)) % 100
