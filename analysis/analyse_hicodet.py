import argparse
import os
import sys

import cv2
import matplotlib
import numpy as np
from matplotlib import pyplot as plt

from analysis.utils import vis_one_image, plot_mat
from config import cfg
from lib.dataset.hicodet.hicodet_split import HicoDetSplit, Splits
from lib.dataset.utils import GTEntry

try:
    matplotlib.use('Qt5Agg')
    # sys.argv[1:] = ['vis']
    # sys.argv[1:] = ['stats']
    sys.argv[1:] = ['find']
except ImportError:
    pass


def stats():
    output_dir = os.path.join('analysis', 'output', 'gt', 'stats')
    split = Splits.TRAIN

    os.makedirs(output_dir, exist_ok=True)
    hds = HicoDetSplit.get_split(split=split)
    hd = hds.hicodet

    op_mat = np.zeros([hds.num_object_classes, hds.num_predicates])
    for _, p, o in hds.hoi_triplets:
        op_mat[o, p] += 1
    pred_labels = hd.predicates
    obj_labels = hd.objects

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
    cfg.parse_args(allow_required=False, reset=True)
    hds = HicoDetSplit.get_split(split=Splits.TRAIN)

    query_str = ['hold', 'refrigerator']
    query = [hds.hicodet._pred_index[query_str[0]], hds.hicodet._obj_class_index[query_str[1]]]

    output_dir = os.path.join('analysis', 'output', 'gt', 'find', '-'.join(query_str))
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(len(hds)):
        example = hds.get_img_entry(idx, read_img=False)  # type: GTEntry
        im_fn = example.filename

        boxes = example.gt_boxes
        box_classes = example.gt_obj_classes
        hois = example.gt_hois
        for _, i, o_ind in hois:
            if i == query[0] and box_classes[o_ind] == query[1]:
                break
        else:  # This is correct!
            continue
        ho_pairs = hois[:, [0, 2]]
        action_class_scores = np.zeros((ho_pairs.shape[0], hds.num_predicates))
        action_class_scores[np.arange(action_class_scores.shape[0]), example.gt_hois[:, 1]] = 1

        im = cv2.imread(os.path.join(hds.img_dir, im_fn))
        vis_one_image(
            hds, im[:, :, [2, 1, 0]],  # BGR -> RGB for visualization
            boxes=boxes, box_classes=box_classes, box_classes_scores=np.ones_like(box_classes), masks=None,
            ho_pairs=ho_pairs, action_class_scores=action_class_scores,
            output_file_path=os.path.join(output_dir, os.path.splitext(im_fn)[0]),
            ext='png',
            dpi=400, fontsize=2, show_scores=False
        )

        # if idx >= 1000:
        #     break


def vis_gt():
    cfg.parse_args(allow_required=False, reset=True)
    hds = HicoDetSplit.get_split(split=Splits.TEST)

    output_dir = os.path.join('analysis', 'output', 'vis', 'gt')
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(len(hds)):
        example = hds.get_img_entry(idx, read_img=False)  # type: GTEntry
        im_fn = example.filename
        # if im_fn not in [s.strip() for s in """
        # HICO_train2015_00001418.jpg
        # """.split('\n')]:
        #     continue

        boxes = example.gt_boxes
        box_classes = example.gt_obj_classes
        ho_pairs = example.gt_hois[:, [0, 2]]
        action_class_scores = np.zeros((ho_pairs.shape[0], hds.num_predicates))
        action_class_scores[np.arange(action_class_scores.shape[0]), example.gt_hois[:, 1]] = 1

        im = cv2.imread(os.path.join(hds.img_dir, im_fn))
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
