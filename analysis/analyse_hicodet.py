import argparse
import os
import pickle
import sys
from typing import List

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from analysis.utils import vis_one_image, plot_mat
from config import cfg
from lib.dataset.hicodet.hicodet import HicoDet, HicoDetImData
from lib.dataset.utils import Splits

try:
    matplotlib.use('Qt5Agg')
    # sys.argv[1:] = ['vis']
    sys.argv[1:] = ['find']
    # sys.argv[1:] = ['stats']
except ImportError:
    pass


def stats():
    output_dir = os.path.join('analysis', 'output', 'gt', 'stats')
    split = Splits.TRAIN

    os.makedirs(output_dir, exist_ok=True)
    hds = HicoDetSplitBuilder.get_split(HicoDetSplit, split=split)  # type: HicoDetSplit
    hd = hds.full_dataset

    op_mat = np.zeros([hds.num_objects, hds.num_actions])
    for _, p, o in hds.hoi_triplets:
        op_mat[o, p] += 1
    pred_labels = hd.actions
    obj_labels = hd.objects

    perc_op_mat = op_mat / op_mat.sum(axis=1, keepdims=True)
    pred_inds = sorted(pickle.load(open('output/base/2019-06-14_10-13-10_pfilt/ds_inds.pkl', 'rb'))[Splits.TRAIN.value]['pred'].tolist())
    for o, row in enumerate(perc_op_mat):
        print(f'{hd.objects[o]:15s}',
              ', '.join([f'{hd.actions[p]:>15s}={row[p] * 100:4.1f}' for p in sorted(set(hd.action_index.values()) - set(pred_inds))]))

    # Sort by most frequent object and predicate
    num_objs_per_predicate = np.sum(op_mat, axis=0)
    pred_inds = (np.argsort(num_objs_per_predicate[1:])[::-1] + 1).tolist() + [0]  # no_interaction at the end
    pred_labels = [pred_labels[i] for i in pred_inds]
    op_mat = op_mat[:, pred_inds]

    num_preds_per_object = np.sum(op_mat, axis=1)
    obj_inds = np.argsort(num_preds_per_object)[::-1]
    obj_labels = [obj_labels[i] for i in obj_inds]
    op_mat = op_mat[obj_inds, :]

    # # Use different colors
    # for i in range(op_mat.shape[0]):
    #     for j in range(op_mat.shape[1]):
    #         op_mat[i, j] -= 0.5 * i / op_mat.shape[0]
    #         op_mat[i, j] -= 0.5 * j / op_mat.shape[1]

    plot_mat(op_mat, pred_labels, obj_labels, x_inds=pred_inds, y_inds=obj_inds, vrange=None, plot=False)
    plt.savefig(os.path.join(output_dir, 'freq_%s.png' % split.value), dpi=300)


def find():
    split = Splits.TEST
    hicodet = HicoDet()
    split_data = hicodet.split_data[split]

    queries_str = [
        # ['cook', 'pizza'],
        # ['eat', 'sandwich'],
        # ['eat', 'apple'],
        # ['stab', 'person'],
        ['hug', 'cat'],
    ]
    queries = [hicodet.oa_pair_to_interaction[hicodet.object_index[q[1]], hicodet.action_index[q[0]]]
               for q in queries_str]
    if np.any(np.array(queries) < 0):
        raise ValueError('Unknown interaction(s).')
    output_dir = os.path.join('analysis', 'output', 'gt', 'find', '_'.join(['-'.join(q) for q in queries_str]))

    os.makedirs(output_dir, exist_ok=True)

    queries_set = set(queries)
    for idx, im_data in enumerate(split_data):
        im_data = im_data  # type: HicoDetImData
        im_fn = im_data.filename

        if idx % 1000 == 0:
            print(idx)

        boxes = im_data.boxes
        box_classes = im_data.box_classes
        hois = im_data.hois
        gt_interactions = hicodet.oa_pair_to_interaction[box_classes[hois[:, 2]], hois[:, 1]]
        misses = queries_set - set(gt_interactions.tolist())
        if misses:
            continue
        ho_pairs = hois[:, [0, 2]]
        action_class_scores = np.zeros((ho_pairs.shape[0], hicodet.num_actions))
        action_class_scores[np.arange(action_class_scores.shape[0]), hois[:, 1]] = 1

        im = cv2.imread(os.path.join(hicodet.get_img_dir(split=split), im_fn))
        vis_one_image(
            hicodet, im[:, :, [2, 1, 0]],  # BGR -> RGB for visualization
            boxes=boxes, box_classes=box_classes, box_classes_scores=np.ones_like(box_classes), masks=None,
            ho_pairs=ho_pairs, action_class_scores=action_class_scores,
            output_file_path=os.path.join(output_dir, os.path.splitext(im_fn)[0]),
            ext='png',
            dpi=400, fontsize=3, show_scores=False
        )

        # if idx >= 1000:
        #     break


def vis_gt():
    cfg.parse_args(fail_if_missing=False, reset=True)
    split = Splits.TRAIN
    hd = HicoDet()
    hds = hd.split_data[split]  # type: List[HicoDetImData]

    output_dir = os.path.join('analysis', 'output', 'vis', 'gt')
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(len(hds)):
        example = hds[idx]
        im_fn = example.filename
        # if im_fn not in [s.strip() for s in """
        # HICO_train2015_00001418.jpg
        # """._data_split('\n')]:
        #     continue

        boxes = example.boxes
        box_classes = example.box_classes
        ho_pairs = example.hois[:, [0, 2]]
        action_class_scores = np.zeros((ho_pairs.shape[0], hd.num_actions))
        action_class_scores[np.arange(action_class_scores.shape[0]), example.hois[:, 1]] = 1

        im = cv2.imread(os.path.join(hd.get_img_dir(split), im_fn))
        vis_one_image(
            hds, im[:, :, [2, 1, 0]],  # BGR -> RGB for visualization
            boxes=boxes, box_classes=box_classes, box_classes_scores=np.ones_like(box_classes), masks=None,
            ho_pairs=ho_pairs, action_class_scores=action_class_scores,
            output_file_path=os.path.join(output_dir, os.path.splitext(im_fn)[0]),
            ext='png',
            dpi=400, fontsize=2, show_scores=False
        )

        if idx >= 1000:
            break


def main():
    funcs = {'vis': vis_gt,
             'stats': stats,
             'find': find,
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
