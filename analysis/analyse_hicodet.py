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
from lib.dataset.utils import Minibatch, get_counts, Example
from lib.knowledge_extractors.imsitu_knowledge_extractor import ImSituKnowledgeExtractor
from lib.models.utils import Prediction
from lib.stats.evaluator_HD import Evaluator as Evaluator_HD
from lib.stats.evaluator import Evaluator, MetricFormatter
from scripts.utils import get_all_models_by_name

try:
    matplotlib.use('Qt5Agg')
    sys.argv[1:] = ['vis']
except ImportError:
    pass


def vis_gt():
    cfg.parse_args(allow_required=False, reset=True)
    hds = HicoDetInstanceSplit.get_split(split=Splits.TEST, load_precomputed=False)

    output_dir = os.path.join('analysis', 'output', 'vis', 'gt')
    os.makedirs(output_dir, exist_ok=True)

    for idx in range(len(hds)):
        example = hds.get_entry(idx, read_img=False, ignore_precomputed=True)  # type: Example
        im_fn = example.fn
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
