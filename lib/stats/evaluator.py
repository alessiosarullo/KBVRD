from collections import Counter
from typing import List, Dict

import pickle
import numpy as np

from lib.bbox_utils import compute_ious
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import Example
from lib.models.utils import Prediction


class BaseEvaluator:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load(self, fn):
        with open(fn, 'rb') as f:
            d = pickle.load(f)
            self.__dict__.update(d)

    def save(self, fn):
        with open(fn, 'wb') as f:
            pickle.dump({k: v for k, v in vars(self).items() if k not in ['dataset']}, f)

    def evaluate_predictions(self, predictions: List[Dict]):
        raise NotImplementedError()


class Evaluator(BaseEvaluator):
    def __init__(self, dataset: HicoDetInstanceSplit, iou_thresh=0.5, hoi_score_thr=None, num_hoi_thr=None):
        super().__init__()
        self.iou_thresh = iou_thresh
        self.hoi_score_thr = hoi_score_thr
        self.num_hoi_thr = num_hoi_thr

        self.dataset = dataset
        self._init()

    def _init(self):
        self.gt_hoi_classes = []
        self.predict_hoi_scores = []
        self.pred_gt_ho_assignment = []
        self.gt_count = 0

        self.metrics = {}  # type: Dict[str, np.ndarray]

    def evaluate_predictions(self, predictions: List[Dict]):
        self._init()
        assert len(predictions) == self.dataset.num_images, (len(predictions), self.dataset.num_images)

        for i, res in enumerate(predictions):
            ex = self.dataset.get_entry(i, read_img=False, ignore_precomputed=True)
            prediction = Prediction.from_dict(res)
            self.process_prediction(i, ex, prediction)
        self.compute_metrics()

    def compute_metrics(self):
        gt_hoi_classes = np.concatenate(self.gt_hoi_classes, axis=0)
        assert self.gt_count == gt_hoi_classes.shape[0]
        predict_hoi_scores = np.concatenate(self.predict_hoi_scores, axis=0)
        pred_gt_ho_assignment = np.concatenate(self.pred_gt_ho_assignment, axis=0)

        gt_hoi_classes_count = Counter(gt_hoi_classes.tolist())

        ap = np.zeros(self.dataset.num_interactions)
        recall = np.zeros(self.dataset.num_interactions)
        for j in range(self.dataset.num_interactions):
            p_hoi_scores = predict_hoi_scores[:, j]
            p_gt_ho_assignment = pred_gt_ho_assignment[:, j]
            if self.hoi_score_thr is not None:
                inds = (p_hoi_scores >= self.hoi_score_thr)
                p_hoi_scores = p_hoi_scores[inds]
                p_gt_ho_assignment = p_gt_ho_assignment[inds]
            if self.num_hoi_thr is not None:
                p_hoi_scores = p_hoi_scores[:self.num_hoi_thr]
                p_gt_ho_assignment = p_gt_ho_assignment[:self.num_hoi_thr]
            rec_j, prec_j, ap_j = self.eval_interactions(p_hoi_scores, p_gt_ho_assignment, gt_hoi_classes_count[j])
            ap[j] = ap_j
            if rec_j.size > 0:
                recall[j] = rec_j[-1]

        self.metrics['M-mAP'] = ap
        self.metrics['M-rec'] = recall

    def print_metrics(self, sort=False):
        mf = MetricFormatter()
        lines = []

        obj_metrics = {k: v for k, v in self.metrics.items() if k.lower().startswith('obj')}
        lines += mf.format_metric_and_gt_lines(self.dataset.obj_labels, metrics=obj_metrics, gt_str='GT objects', sort=sort)

        hoi_triplets = self.dataset.hoi_triplets
        hois = self.dataset.op_pair_to_interaction[hoi_triplets[:, 2], hoi_triplets[:, 1]]
        assert np.all(hois >= 0)
        hoi_metrics = {k: v for k, v in self.metrics.items() if not k.lower().startswith('obj')}
        lines += mf.format_metric_and_gt_lines(hois, metrics=hoi_metrics, gt_str='GT HOIs', sort=sort)

        printstr = '\n'.join(lines)
        print(printstr)
        return printstr

    def process_prediction(self, im_id, gt_entry: Example, prediction: Prediction):
        if isinstance(gt_entry, Example):
            gt_hoi_triplets = gt_entry.gt_hois[:, [0, 2, 1]]  # (h, o, i)
            num_gt_hois = gt_hoi_triplets.shape[0]

            gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)

            gt_hoi_classes = self.dataset.op_pair_to_interaction[gt_entry.gt_obj_classes[gt_hoi_triplets[:, 1]], gt_hoi_triplets[:, 2]]
            assert np.all(gt_hoi_classes) >= 0

            gt_ho_ids = self.gt_count + np.arange(num_gt_hois)
            self.gt_count += num_gt_hois
        else:
            raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))

        predict_hoi_scores = np.zeros([0, self.dataset.num_interactions])
        predict_ho_pairs = np.zeros((0, 2), dtype=np.int)
        predict_boxes = np.zeros((0, 4))
        if prediction.obj_boxes is not None:
            assert prediction.obj_im_inds.shape[0] == prediction.obj_boxes.shape[0]

            predict_boxes = prediction.obj_boxes

            if prediction.ho_pairs is not None:
                assert len(np.unique(prediction.obj_im_inds)) == len(np.unique(prediction.ho_img_inds)) == 1

                predict_ho_pairs = prediction.ho_pairs
                predict_hoi_scores = prediction.hoi_scores
                if predict_hoi_scores is None:
                    assert prediction.obj_im_inds.shape[0] == prediction.obj_scores.shape[0]
                    assert prediction.action_score_distributions is not None
                    predict_action_scores = prediction.action_score_distributions
                    predict_obj_scores_per_ho_pair = prediction.obj_scores[predict_ho_pairs[:, 1], :]

                    predict_hoi_scores = np.empty([predict_ho_pairs.shape[0], self.dataset.num_interactions])
                    for iid, (pid, oid) in enumerate(self.dataset.interactions):
                        predict_hoi_scores[:, iid] = predict_obj_scores_per_ho_pair[:, oid] * predict_action_scores[:, pid]
        else:
            assert prediction.ho_pairs is None

        pred_gt_ious = compute_ious(predict_boxes, gt_boxes)
        pred_gt_assignment_per_hoi = np.full((predict_hoi_scores.shape[0], self.dataset.num_interactions), fill_value=-1, dtype=np.int)
        for predict_idx, (ph, po) in enumerate(predict_ho_pairs):
            gt_pair_ious = np.zeros(num_gt_hois)
            for gtidx, (gh, go, gi) in enumerate(gt_hoi_triplets):
                iou_h = pred_gt_ious[ph, gh]
                iou_o = pred_gt_ious[po, go]
                gt_pair_ious[gtidx] = min(iou_h, iou_o)
            if np.any(gt_pair_ious >= self.iou_thresh):
                gt_pair_ious_per_hoi = np.zeros((num_gt_hois, self.dataset.num_interactions))
                gt_pair_ious_per_hoi[np.arange(num_gt_hois), gt_hoi_classes] = gt_pair_ious
                gt_assignments = gt_pair_ious_per_hoi.argmax(axis=0)[np.any(gt_pair_ious_per_hoi >= self.iou_thresh, axis=0)]
                gt_hoi_assignments = gt_hoi_classes[gt_assignments]
                assert np.unique(gt_assignments).size == gt_assignments.size
                assert np.unique(gt_hoi_assignments).size == gt_hoi_assignments.size
                pred_gt_assignment_per_hoi[predict_idx, gt_hoi_assignments] = gt_ho_ids[gt_assignments]

        self.gt_hoi_classes.append(gt_hoi_classes)
        self.predict_hoi_scores.append(predict_hoi_scores)
        self.pred_gt_ho_assignment.append(pred_gt_assignment_per_hoi)

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

        pad = len(gt_str)
        if metrics:
            pad = max(pad, max([len(k) for k in metrics.keys()]))

        for k, v in metrics.items():
            assert v.size == 1 or len(inds) == v.size
            lines += [self.format_metric(k, v[inds] if v.size > 1 else v, pad)]
        format_str = '%{}s %8s [%s]'.format(pad + 1)
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
