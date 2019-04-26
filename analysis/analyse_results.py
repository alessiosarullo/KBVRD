import argparse
import os
import pickle
import sys
from collections import Counter
from typing import Dict

import cv2
import matplotlib
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib import pyplot as plt

from analysis.utils import vis_one_image, plot_mat, heatmap, annotate_heatmap
from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit, Splits
from lib.dataset.utils import Minibatch, get_counts
from lib.knowledge_extractors.imsitu_knowledge_extractor import ImSituKnowledgeExtractor
from lib.models.utils import Prediction
from lib.stats.evaluator_HD import Evaluator as Evaluator_HD
from lib.stats.evaluator import Evaluator, MetricFormatter
from scripts.utils import get_all_models_by_name

try:
    matplotlib.use('Qt5Agg')
    # sys.argv[1:] = ['eval', '--save_dir', 'output/inter/2019-04-23_16-38-16_vanilla']
    sys.argv[1:] = ['eval', '--save_dir', 'output/hoi/2019-04-22_16-06-15_dbatch-32']
    # sys.argv[1:] = ['vis', '--save_dir', 'output/hoi/2019-04-22_17-04-13_b64']
except ImportError:
    pass


def _setup_and_load():
    cfg.parse_args(allow_required=False, reset=True)

    with open(cfg.program.result_file, 'rb') as f:
        results = pickle.load(f)
    cfg.load()
    return results


def evaluate():
    results = _setup_and_load()
    hds = HicoDetInstanceSplit.get_split(split=Splits.TEST)
    evaluator = Evaluator(dataset=hds, hoi_score_thr=None, num_hoi_thr=None)
    stats = evaluator.evaluate_predictions(results)
    # stats = Evaluator_HD.evaluate_predictions(hds, results)
    stats.print_metrics(sort=True)


# def stats():
#     base_argv = sys.argv
#     exps = [
#         'output/hoi/2019-04-12_09-51-28_red-ored',
#         # 'output/zero/2019-04-03_10-28-10_vanilla',
#         # 'output/kb/2019-04-09_11-47-18_ds-imsitu'
#     ]
#     true_pos = []
#     pos = None
#     for exp in exps:
#         sys.argv = base_argv + ['--save_dir', exp]
#         print('=' * 100, '\n', sys.argv)
#
#         results = _setup_and_load()
#
#         hdtrain = HicoDetInstanceSplit.get_split(split=Splits.TRAIN)
#         hdtest = HicoDetInstanceSplit.get_split(split=Splits.TEST)
#
#         stats = Evaluator_old.evaluate_predictions(hdtest, results)  # type: Evaluator_old
#         stats.print_metrics(sort=True)
#
#         # detector = get_all_models_by_name()[cfg.program.model](hdtrain)
#         # ckpt = torch.load(cfg.program.saved_model_file, map_location='cpu')
#         # detector.load_state_dict(ckpt['state_dict'])
#
#         # op_adj_mat = detector.hoi_branch.op_adj_mat.squeeze(dim=-1).detach().numpy()
#         # op_conf_mat = torch.sigmoid(detector.hoi_branch.op_conf_mat.squeeze(dim=-1).detach()).numpy()
#         #
#         # ds_counts = get_counts(dataset=hdtrain)
#         # ds_counts[:, 0] = 0  # exclude null interaction
#         # ds = np.minimum(1, ds_counts)
#         # imsitu_counts = ImSituKnowledgeExtractor().extract_freq_matrix(hdtrain)
#         # imsitu_counts[:, 0] = 0  # exclude null interaction
#         # imsitu = np.minimum(1, imsitu_counts)  # only check if the pair exists (>=1 occurrence) or not (0 occurrences)
#
#         # # Plot
#         # plt.figure(figsize=(16, 9))
#
#         # gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1],
#         #                        wspace=0.01, hspace=0.4, top=0.9, bottom=0.1, left=0.05, right=0.95)
#         # plot_mat(op_adj_mat[:, :, 1] / 3 + ds * 2 / 3, predicates, objects, axes=plt.subplot(gs[0, 0]))
#         # plot_mat(op_conf_mat[:, :, 1], predicates, objects, axes=plt.subplot(gs[0, 1]))
#         # plot_mat(op_adj_mat[:, :, 0] / 3 + imsitu * 2 / 3, predicates, objects, axes=plt.subplot(gs[1, 0]))
#         # plot_mat(op_conf_mat[:, :, 0], predicates, objects, axes=plt.subplot(gs[1, 1]))
#         # plt.show()
#
#         pred_thr = 0.5
#         gt_hois = np.concatenate(stats.hoi_labels, axis=0)
#         bg_hois = gt_hois[:, 0] > 0
#         fg_hois = np.any(gt_hois[:, 1:], axis=1)
#         assert not np.any(bg_hois & fg_hois), np.flatnonzero(bg_hois & fg_hois)
#         gt_hoi_objs = np.concatenate(stats.hoi_obj_labels, axis=0)
#         predictions = np.concatenate(stats.hoi_predictions, axis=0)
#         assert gt_hois.shape[0] == gt_hoi_objs.shape[0] == predictions.shape[0] and  \
#               gt_hois.shape[1] == predictions.shape[1] == hdtrain.num_predicates
#
#         tps = np.zeros([hdtrain.num_object_classes, hdtrain.num_predicates])
#         gt_counts = np.zeros_like(tps)
#         confmat = np.zeros([hdtrain.num_predicates, hdtrain.num_predicates])
#         confmat_norm_factor = np.zeros(hdtrain.num_predicates)
#         for gth_1hot, gto, ph_1hot in zip(gt_hois, gt_hoi_objs, predictions):
#             gthois = np.flatnonzero(gth_1hot)
#             tps[gto, gthois] += (ph_1hot[gthois] > pred_thr)
#             gt_counts[gto, gthois] += 1
#
#             gthois = set(gthois.tolist())
#             phois = set(np.flatnonzero(ph_1hot > pred_thr))
#             hits = np.array(sorted(gthois & phois))
#             unmatched = gthois - phois
#             mistaken_for = np.array(sorted(phois - gthois))
#             if hits.size > 0:
#                 confmat[hits, hits] += 1
#                 confmat_norm_factor[hits] += 1
#             for u in unmatched:
#                 if mistaken_for.size > 0:
#                     confmat[u, mistaken_for] += 1
#                 else:  # assign to no_interaction by default
#                     confmat[u, 0] += 1
#
#         assert np.all(tps <= gt_counts)
#         if pos is None:
#             pos = gt_counts
#         assert np.all(pos == gt_counts)
#         true_pos.append(tps)
#
#         # assert np.allclose(confmat.sum(axis=1), gt_counts.sum(axis=0))  # This is not true because a HOI can be mistaken for several other ones
#         confmat /= confmat.sum(axis=1, keepdims=True)
#
#     obj_inds = np.argsort(pos.sum(axis=1))[::-1]
#     # pred_inds = np.argsort(pos.sum(axis=0))[::-1]
#     pred_inds = np.array((np.argsort(pos.sum(axis=0)[1:])[::-1] + 1).tolist() + [0])  # no_interaction at the end
#
#     print(MetricFormatter().format_metric('01 loss', np.diag(confmat)[pred_inds]))
#
#     plot_mat(confmat[:, pred_inds][pred_inds, :], [hdtrain.predicates[i] for i in pred_inds], [hdtrain.predicates[i] for i in pred_inds],
#              x_inds=pred_inds, y_inds=pred_inds,
#              cbar=True, bin_colours=False, plot=False, grid=False)
#
#     # true_pos = np.stack(true_pos, axis=2)
#     # assert true_pos.shape[2] == 2
#     # mat = (true_pos[:, :, 0] - true_pos[:, :, 1]) / pos
#     # mat = mat / 2 + 0.5
#     # plot_mat(mat[:, pred_inds][obj_inds, :], [hdtrain.predicates[i] for i in pred_inds], [hdtrain.objects[i] for i in obj_inds],
#     #          x_inds=pred_inds, y_inds=obj_inds,
#     #          cbar=True, bin_colours=True, plot=False)
#
#     plt.show()


def vis_masks():
    results = _setup_and_load()
    hds = HicoDetInstanceSplit.get_split(split=Splits.TEST)
    hdsl = hds.get_loader(batch_size=1, shuffle=False)

    output_dir = os.path.join('analysis', 'output', 'vis', *(cfg.program.output_path.split('/')[1:]))
    os.makedirs(output_dir, exist_ok=True)
    dataset_classes = hds.objects

    for b_idx, example in enumerate(hdsl):
        example = example  # type: Minibatch
        im_fn = example.other_ex_data[0]['fn']
        if im_fn not in [s.strip() for s in """
        HICO_test2015_00000648.jpg
        """.split('\n')]:
            continue

        prediction_dict = results[b_idx]
        prediction = Prediction.from_dict(prediction_dict)

        boxes = prediction.obj_boxes
        obj_scores = prediction.obj_scores
        box_classes = np.argmax(obj_scores, axis=1)
        box_class_scores = obj_scores[np.arange(boxes.shape[0]), box_classes]
        print(boxes)
        print(box_classes)
        print(box_class_scores)

        boxes_with_scores = np.concatenate([boxes, box_class_scores[:, None]], axis=1)
        im = cv2.imread(os.path.join(hds.img_dir, im_fn))
        vis_one_image(
            im[:, :, [2, 1, 0]],  # BGR -> RGB for visualization
            boxes=boxes_with_scores,
            box_classes=box_classes,
            class_names=dataset_classes,
            masks=None,
            output_file_path=os.path.join(output_dir, os.path.splitext(im_fn)[0]),
            box_alpha=0.3,
            show_class=True,
            thresh=0.0,  # Lower this to see all the predictions (was 0.7 in the original code)
            ext='png'
        )
        # break


def main():
    funcs = {'vis': vis_masks,
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
