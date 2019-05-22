import argparse
import os
import pickle
import sys
from typing import List, Dict

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from analysis.utils import vis_one_image, plot_mat
from config import cfg
from lib.bbox_utils import compute_ious
from lib.dataset.hicodet import HicoDetInstanceSplit, Splits
from lib.dataset.utils import Minibatch, Example
from lib.models.utils import Prediction
from lib.stats.evaluator import Evaluator
from lib.stats.utils import Timer

try:
    matplotlib.use('Qt5Agg')
    sys.argv[1:] = ['eval', '--save_dir', 'output/actonly/2019-05-13_13-39-01_vanilla']
    # sys.argv[1:] = ['stats', '--save_dir', 'output/actonly/2019-05-13_13-39-01_vanilla']
    # sys.argv[1:] = ['vis', '--save_dir', 'output/actonly/2019-05-13_13-39-01_vanilla']
except ImportError:
    pass


class Analyser:
    def __init__(self, dataset: HicoDetInstanceSplit, iou_thresh=0.5, hoi_score_thr=None, num_hoi_thr=None):
        super().__init__()
        self.iou_thresh = iou_thresh
        self.hoi_score_thr = hoi_score_thr
        self.num_hoi_thr = num_hoi_thr

        self.dataset = dataset
        self.gt_interactions = []  # (p, o)
        self.predict_ho_obj_scores = []
        self.predict_action_scores = []
        self.predict_gt_assignment_unconstrained = []
        self.gt_count = 0

    def get_stats(self, predictions: List[Dict]):
        assert len(predictions) == self.dataset.num_images, (len(predictions), self.dataset.num_images)

        for i, res in enumerate(predictions):
            ex = self.dataset.get_entry(i, read_img=False, ignore_precomputed=True)
            prediction = Prediction.from_dict(res)
            self.process_prediction(i, ex, prediction)

        return self.get_stats_for_spatial_matches()

    def process_prediction(self, im_id, gt_entry: Example, prediction: Prediction):
        if isinstance(gt_entry, Example):
            gt_hoi_triplets = gt_entry.gt_hois[:, [0, 2, 1]]  # (h, o, i)
            num_gt_hois = gt_hoi_triplets.shape[0]

            gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)

            gt_ho_ids = self.gt_count + np.arange(num_gt_hois)
            self.gt_count += num_gt_hois
        else:
            raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))

        predict_action_scores = np.zeros([0, self.dataset.num_predicates])
        predict_obj_scores_per_ho_pair = np.zeros([0, self.dataset.num_object_classes])
        predict_ho_pairs = np.zeros((0, 2), dtype=np.int)
        predict_boxes = np.zeros((0, 4))
        if prediction.obj_boxes is not None:
            assert prediction.obj_im_inds.shape[0] == prediction.obj_boxes.shape[0]

            predict_boxes = prediction.obj_boxes

            if prediction.ho_pairs is not None:
                assert len(np.unique(prediction.obj_im_inds)) == len(np.unique(prediction.ho_img_inds)) == 1

                predict_ho_pairs = prediction.ho_pairs
                assert prediction.hoi_scores is None
                assert prediction.obj_im_inds.shape[0] == prediction.obj_scores.shape[0]
                assert prediction.action_score_distributions is not None

                predict_action_scores = prediction.action_score_distributions
                predict_obj_scores_per_ho_pair = prediction.obj_scores[predict_ho_pairs[:, 1], :]
        else:
            assert prediction.ho_pairs is None

        pred_gt_ious = compute_ious(predict_boxes, gt_boxes)
        pred_gt_assignment = np.full(predict_ho_pairs.shape[0], fill_value=-1, dtype=np.int)
        for predict_idx, (ph, po) in enumerate(predict_ho_pairs):
            gt_pair_ious = np.zeros(num_gt_hois)
            for gtidx, (gh, go, gi) in enumerate(gt_hoi_triplets):
                iou_h = pred_gt_ious[ph, gh]
                iou_o = pred_gt_ious[po, go]
                gt_pair_ious[gtidx] = min(iou_h, iou_o)
            if np.any(gt_pair_ious >= self.iou_thresh):
                gt_assignment = np.argmax(gt_pair_ious)
                pred_gt_assignment[predict_idx] = gt_ho_ids[gt_assignment]

        self.gt_interactions.append(np.stack([gt_hoi_triplets[:, 2], gt_entry.gt_obj_classes[gt_hoi_triplets[:, 1]]], axis=1))
        self.predict_ho_obj_scores.append(predict_obj_scores_per_ho_pair)
        self.predict_action_scores.append(predict_action_scores)
        self.predict_gt_assignment_unconstrained.append(pred_gt_assignment)

    def get_stats_for_spatial_matches(self):
        gt_interactions = np.concatenate(self.gt_interactions, axis=0)
        predict_ho_obj_scores = np.concatenate(self.predict_ho_obj_scores, axis=0)
        predict_action_scores = np.concatenate(self.predict_action_scores, axis=0)
        pred_gt_assignment_unconstrained = np.concatenate(self.predict_gt_assignment_unconstrained, axis=0)

        assert predict_action_scores.shape[0] == predict_ho_obj_scores.shape[0] == pred_gt_assignment_unconstrained.shape[0]

        tp = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))
        fp = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))
        misses = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))
        num_gt = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))
        num_pred = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))

        for gt_id, o_scores, a_scores in zip(pred_gt_assignment_unconstrained, predict_ho_obj_scores, predict_action_scores):
            if gt_id < 0:
                continue

            po = np.argmax(o_scores)
            pa = np.argmax(a_scores)
            num_pred[po, pa] += 1

            ga, go = gt_interactions[gt_id, :]
            num_gt[go, ga] += 1

            if po == go and pa == ga:
                tp[po, pa] += 1
            else:
                fp[po, pa] += 1
                misses[go, ga] += 1

        return tp, fp, misses, num_gt, num_pred


def _setup_and_load():
    cfg.parse_args(allow_required=False, reset=True)

    with open(cfg.program.prediction_file, 'rb') as f:
        results = pickle.load(f)
    cfg.load()
    return results


def evaluate():
    results = _setup_and_load()
    hds = HicoDetInstanceSplit.get_split(split=Splits.TEST, load_precomputed=False)
    evaluator = Evaluator(dataset=hds, hoi_score_thr=None, num_hoi_thr=None)
    evaluator.evaluate_predictions(results)
    evaluator.print_metrics(sort=True)
    evaluator.save(cfg.program.eval_res_file)
    # stats = Evaluator_HD.evaluate_predictions(hds, results)
    # stats.print_metrics(sort=True)
    Timer.print()


def stats():
    results = _setup_and_load()

    hdtest = HicoDetInstanceSplit.get_split(split=Splits.TEST)

    analyser = Analyser(dataset=hdtest, hoi_score_thr=None, num_hoi_thr=None)
    tp, fp, misses, num_gt, num_pred = analyser.get_stats(results)

    # obj_inds = np.argsort(pos.sum(axis=1))[::-1]
    # pred_inds = np.array((np.argsort(pos.sum(axis=0)[1:])[::-1] + 1).tolist() + [0])  # no_interaction at the end

    # plot_mat(tp / np.maximum(1, misses), hdtest.predicates, hdtest.objects, vrange=None, plot=False)
    x = (num_gt == 0).astype(np.float) * num_pred
    x[x == 0] = np.inf
    plot_mat(x, hdtest.predicates, hdtest.objects, vrange=None, plot=False)

    print('\n'.join(['%-20s %s' % (hdtest.predicates[p], hdtest.objects[o]) for p, o in np.stack(np.where(~np.isinf(x.T)), axis=1)]))

    # TODO save

    plt.show()


def visualise_images():
    act_thr = 0.5

    results = _setup_and_load()
    hds = HicoDetInstanceSplit.get_split(split=Splits.TEST)
    hdsl = hds.get_loader(batch_size=1, shuffle=False)

    output_dir = os.path.join('analysis', 'output', 'vis', *(cfg.program.output_path.split('/')[1:]))
    os.makedirs(output_dir, exist_ok=True)

    for b_idx, example in enumerate(hdsl):
        example = example  # type: Minibatch
        im_fn = example.other_ex_data[0]['fn']
        # if im_fn not in [s.strip() for s in """
        # HICO_test2015_00000648.jpg
        # """.split('\n')]:
        #     continue

        prediction_dict = results[b_idx]
        prediction = Prediction.from_dict(prediction_dict)

        boxes = prediction.obj_boxes
        obj_scores = prediction.obj_scores
        ho_pairs = prediction.ho_pairs
        act_scores = prediction.action_score_distributions
        if obj_scores is None:
            continue

        box_classes = np.argmax(obj_scores, axis=1)
        box_class_scores = obj_scores[np.arange(boxes.shape[0]), box_classes]

        im = cv2.imread(os.path.join(hds.img_dir, im_fn))
        vis_one_image(
            hds, im[:, :, [2, 1, 0]],  # BGR -> RGB for visualization
            boxes=boxes, box_classes=box_classes, box_classes_scores=box_class_scores, masks=None,
            ho_pairs=ho_pairs, action_class_scores=act_scores,
            output_file_path=os.path.join(output_dir, os.path.splitext(im_fn)[0]),
            ext='png',
            act_thr=act_thr,
            dpi=400, fontsize=2,
        )

        if b_idx >= 1000:
            break


def main():
    funcs = {'vis': visualise_images,
             'stats': stats,
             'eval': evaluate,
             }
    print(sys.argv)
    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str, choices=funcs.keys())
    namespace = parser.parse_known_args()
    func = vars(namespace[0])['func']
    sys.argv = sys.argv[:1] + namespace[1]
    print(sys.argv)
    funcs[func]()


if __name__ == '__main__':
    main()
