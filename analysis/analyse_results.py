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
from lib.dataset.hicodet.hicodet_split import HicoDetSplitBuilder, HicoDetSplit, Splits, Example, Minibatch
from lib.models.containers import Prediction
from lib.stats.evaluator import Evaluator
from lib.stats.utils import Timer

try:
    matplotlib.use('Qt5Agg')
    sys.argv[1:] = ['eval', '--save_dir', 'output/base/2019-06-05_17-43-04_vanilla']
    # sys.argv[1:] = ['stats', '--save_dir', 'output/base/2019-06-05_17-43-04_vanilla']
    # sys.argv[1:] = ['vis', '--save_dir', 'output/base/2019-06-05_17-43-04_vanilla']
except ImportError:
    pass


class Analyser:
    def __init__(self, dataset: HicoDetSplit, iou_thresh=0.5, hoi_score_thr=0.5):
        super().__init__()

        self.iou_thresh = iou_thresh
        self.hoi_score_thr = hoi_score_thr

        self.dataset = dataset
        # self.gt_interactions = []  # (p, o)
        # self.predict_ho_obj_scores = []
        # self.predict_action_scores = []
        # self.predict_gt_assignment_unconstrained = []

        # NOTE: this sums up to the number of predictions.
        # Actually, it does not. I am not considering predictions that do not match any GT entry because of the sparsity of annotations. If
        # required, those should be considered mistakes of the null interaction.
        self.act_conf_mat = np.zeros((self.dataset.num_predicates, self.dataset.num_predicates))

        self.gt_matches = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))
        self.gt_spatial_matches = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))
        self.gt_candidate_matches = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))
        self.num_gt = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))
        self.num_pred = np.zeros((self.dataset.num_object_classes, self.dataset.num_predicates))

    def compute_stats(self, predictions: List[Dict]):
        assert len(predictions) == self.dataset.num_images, (len(predictions), self.dataset.num_images)

        for i, res in enumerate(predictions):
            ex = self.dataset.get_img_entry(i, read_img=False)
            prediction = Prediction(res)
            self.process_prediction(i, ex, prediction)

        self.act_conf_mat /= np.maximum(1, self.act_conf_mat.sum(axis=1))

    def process_prediction(self, im_id, gt_entry: Example, prediction: Prediction):
        if isinstance(gt_entry, Example):
            gt_hoi_triplets = gt_entry.gt_hois[:, [0, 2, 1]]  # (h, o, i)
            gt_interactions = np.stack([gt_hoi_triplets[:, 2], gt_entry.gt_obj_classes[gt_hoi_triplets[:, 1]]], axis=1)  # (a, o)
            num_gt_hois = gt_hoi_triplets.shape[0]

            gt_boxes = gt_entry.gt_boxes.astype(np.float, copy=False)
        else:
            raise ValueError('Unknown type for GT entry: %s.' % str(type(gt_entry)))

        predict_action_scores = np.zeros([0, self.dataset.num_predicates])
        predict_ho_obj_scores = np.zeros([0, self.dataset.num_object_classes])
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
                assert prediction.action_scores is not None

                predict_action_scores = prediction.action_scores
                predict_ho_obj_scores = prediction.obj_scores[predict_ho_pairs[:, 1], :]
        else:
            assert prediction.ho_pairs is None

        num_predictions = predict_ho_pairs.shape[0]

        pred_gt_ious = compute_ious(predict_boxes, gt_boxes)
        pred_ho_objects = np.argmax(predict_ho_obj_scores, axis=1)

        # First, find predictions that 1) match spatially with GT and 2) classify the object correctly.
        pred_gt_spatial_matches = np.zeros((num_predictions, num_gt_hois), dtype=bool)
        pred_gt_candidate_matches = np.zeros((num_predictions, num_gt_hois))
        for predict_idx, (ph_ind, po_ind) in enumerate(predict_ho_pairs):
            gt_pair_ious = np.zeros(num_gt_hois)
            for gtidx, (gh_ind, go_ind, ga) in enumerate(gt_hoi_triplets):
                iou_h = pred_gt_ious[ph_ind, gh_ind]
                iou_o = pred_gt_ious[po_ind, go_ind]
                gt_pair_ious[gtidx] = min(iou_h, iou_o)

            spatial_matches = (gt_pair_ious >= self.iou_thresh)
            po = pred_ho_objects[predict_idx]
            matches = np.flatnonzero((gt_interactions[:, 1] == po) & spatial_matches)

            pred_gt_spatial_matches[predict_idx, spatial_matches] = 1
            pred_gt_candidate_matches[predict_idx, matches] = gt_pair_ious[matches]

            self.num_pred[po, predict_action_scores[predict_idx, :] >= self.hoi_score_thr] += 1

        # Then decide which action predictions are hits. For each pair, an action is considered predicted if the assigned score satisfies some
        # criterion (e.g., is greater than a certain threshold). An action prediction is a hit if there is a matching GT (in the sense defined above)
        # with the same action (note: duplicates are NOT allowed, i.e., only the best match for a GT elements is considered correct).
        hits = np.zeros_like(predict_action_scores)
        best_match_for_gt = np.full(num_gt_hois, fill_value=-1, dtype=np.int)
        for gtidx, (ga, go) in enumerate(gt_interactions):
            self.num_gt[go, ga] += 1

            if np.any(pred_gt_spatial_matches[:, gtidx]):
                self.gt_spatial_matches[go, ga] += 1

            pred_candidate_ious = pred_gt_candidate_matches[:, gtidx]
            pred_candidates = (pred_candidate_ious > 0)

            if np.any(pred_candidates):
                self.gt_candidate_matches[go, ga] += 1

                best_match = np.argmax(pred_candidate_ious)
                best_match_for_gt[gtidx] = best_match
                if predict_action_scores[best_match, ga] >= self.hoi_score_thr:
                    hits[best_match, ga] += 1
                    self.gt_matches[go, ga] += 1
                    self.act_conf_mat[ga, ga] += 1
        hits = (hits > 0)

        # Finally, fill the off-diagonal elements of the confusion matrix. To do this, find the best match for any given GT triplet and consider
        # false positives all its predicted actions that are not hits (see above). Another for loop is required because the GT is not multi-label.
        for gtidx, (ga, go) in enumerate(gt_interactions):
            best_match = best_match_for_gt[gtidx]
            if best_match >= 0:
                predicted_acts = (predict_action_scores[best_match, :] >= self.hoi_score_thr)
                if ~hits[best_match, ga]:
                    prediction_misses = predicted_acts & ~hits[best_match, :]
                    assert not prediction_misses[ga]
                    self.act_conf_mat[ga, prediction_misses] += 1

        # # Uncomment this if it is desired to considered unmatched detections as misses of the null interactions.
        # matches_set = set(np.flatnonzero(best_match_for_gt >= 0).tolist())
        # for predict_idx in range(num_predictions):
        #     if predict_idx not in matches_set:
        #         assert not np.any(hits[predict_idx, :])
        #         predicted_acts = (predict_action_scores[predict_idx, :] >= self.hoi_score_thr)
        #         self.act_conf_mat[0, predicted_acts] += 1


def _setup_and_load():
    cfg.parse_args(fail_if_missing=False, reset=True)

    with open(cfg.program.prediction_file, 'rb') as f:
        results = pickle.load(f)
    cfg.load()
    return results


def evaluate():
    results = _setup_and_load()
    hds = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TEST)
    evaluator = Evaluator(dataset_split=hds, hoi_score_thr=None, num_hoi_thr=None)

    # evaluator.evaluate_predictions(results)
    evaluator.load(cfg.program.eval_res_file)

    evaluator.output_metrics(sort=True, actions_to_keep=[1, 2, 84])
    # evaluator.output_metrics(sort=True, actions_to_keep=list(set(range(117)) - {84}))
    # evaluator.output_metrics(sort=True, actions_to_keep=list(range(117)))

    # stats = Evaluator_HD.evaluate_predictions(hds, results)
    # stats.print_metrics(sort=True)
    # Timer.print()


def stats():
    results = _setup_and_load()
    res_save_path = cfg.program.res_stats_path
    os.makedirs(res_save_path, exist_ok=True)

    hdtest = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TEST)

    analyser = Analyser(dataset=hdtest)

    analyser.compute_stats(results)
    num_gt, num_pred = analyser.num_gt, analyser.num_pred
    recall = analyser.gt_matches / num_gt
    ov_obj_recall = analyser.gt_candidate_matches / num_gt
    ov_recall = analyser.gt_spatial_matches / num_gt
    act_conf_mat = analyser.act_conf_mat

    obj_inds = np.argsort(num_gt.sum(axis=1))[::-1]
    s_objects = [hdtest.objects[i] for i in obj_inds]
    pred_inds = (np.argsort(num_gt.sum(axis=0)[1:])[::-1] + 1).tolist() + [0]  # no_interaction at the end
    s_predicates = [hdtest.predicates[i] for i in pred_inds]

    print(np.mean(recall[num_gt > 0]))

    zero_shot_preds = (num_gt == 0).astype(np.float) * num_pred
    zero_shot_preds[zero_shot_preds == 0] = np.inf
    plot_mat(zero_shot_preds, hdtest.predicates, hdtest.objects, vrange=None, plot=False)
    plt.savefig(os.path.join(res_save_path, 'zero_shot.png'), dpi=300)
    zero_shot_str = '\n'.join(['%-20s %-20s %d' % (hdtest.predicates[p], hdtest.objects[o], zero_shot_preds[o, p])
                               for p, o in np.stack(np.where(~np.isinf(zero_shot_preds.T)), axis=1)])
    print(zero_shot_str)
    with open(os.path.join(res_save_path, 'zero_shot.txt'), 'w') as f:
        f.write(zero_shot_str)

    plot_mat(act_conf_mat[pred_inds, :][:, pred_inds], s_predicates, s_predicates, x_inds=pred_inds, y_inds=pred_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'conf_mat.png'), dpi=300)

    plot_mat(recall[obj_inds, :][:, pred_inds], s_predicates, s_objects, x_inds=pred_inds, y_inds=obj_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'matches.png'), dpi=300)

    plot_mat(ov_obj_recall[obj_inds, :][:, pred_inds], s_predicates, s_objects, x_inds=pred_inds, y_inds=obj_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'ov-obj_matches.png'), dpi=300)

    plot_mat(ov_recall[obj_inds, :][:, pred_inds], s_predicates, s_objects, x_inds=pred_inds, y_inds=obj_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'ov_matches.png'), dpi=300)

    plot_mat((1 - recall)[obj_inds, :][:, pred_inds], s_predicates, s_objects, x_inds=pred_inds, y_inds=obj_inds, plot=False)
    plt.savefig(os.path.join(res_save_path, 'misses.png'), dpi=300)

    plt.show()


def visualise_images():
    act_thr = 0.1

    results = _setup_and_load()
    hds = HicoDetSplitBuilder.get_split(HicoDetSplit, split=Splits.TEST)

    output_dir = os.path.join('analysis', 'output', 'vis', *(cfg.program.output_path.split('/')[1:]))
    os.makedirs(output_dir, exist_ok=True)

    for b_idx in range(len(hds)):
        entry = hds.get_img_entry(b_idx, read_img=False)  # type: Example
        im_fn = entry.filename

        prediction_dict = results[b_idx]
        prediction = Prediction(prediction_dict)

        boxes = prediction.obj_boxes
        obj_scores = prediction.obj_scores
        ho_pairs = prediction.ho_pairs
        act_scores = prediction.action_scores
        if obj_scores is None:
            continue

        box_classes = np.argmax(obj_scores, axis=1)
        box_class_scores = obj_scores[np.arange(boxes.shape[0]), box_classes]

        # if im_fn not in [s.strip() for s in """
        # HICO_test2015_00000003.jpg
        # """.split('\n')]:
        #     continue

        # FIND
        # if act_scores[box_classes[ho_pairs[:, 1]] == 18, 9].size == 0 or np.all(act_scores[box_classes[ho_pairs[:, 1]] == 18, 9] < act_thr):
        #     continue
        # print(im_fn)

        im = cv2.imread(os.path.join(hds.img_dir, im_fn))
        assert np.all(boxes[:, [0, 2]] < im.shape[1]) and np.all(boxes[:, [1, 3]] < im.shape[0]), b_idx
        continue
        vis_one_image(
            hds, im[:, :, [2, 1, 0]],  # BGR -> RGB for visualization
            boxes=boxes, box_classes=box_classes, box_classes_scores=box_class_scores, masks=None,
            ho_pairs=ho_pairs, action_class_scores=act_scores,
            output_file_path=os.path.join(output_dir, os.path.splitext(im_fn)[0]),
            ext='png',
            act_thr=act_thr,
            dpi=400, fontsize=2,
        )

        if b_idx >= 0:
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
