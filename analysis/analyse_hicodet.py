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


def vis_masks():
    cfg.parse_args(allow_required=False, reset=True)
    hds = HicoDetInstanceSplit.get_split(split=Splits.TRAIN, load_precomputed=False)

    output_dir = os.path.join('analysis', 'output', 'vis', 'gt')
    os.makedirs(output_dir, exist_ok=True)
    dataset_classes = hds.objects

    for idx in range(len(hds)):
        example = hds.get_entry(idx, read_img=False, ignore_precomputed=True)  # type: Example
        im_fn = example.fn
        if im_fn not in [s.strip() for s in """
        HICO_train2015_00001418.jpg
        """.split('\n')]:
            continue

        boxes = example.gt_boxes
        box_classes = example.gt_obj_classes
        print(boxes)
        print(box_classes)

        boxes_with_scores = np.concatenate([boxes, np.ones((boxes.shape[0], 1))], axis=1)
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
            thresh=0.0,
            ext='png'
        )
        # break


def main():
    funcs = {'vis': vis_masks,
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
