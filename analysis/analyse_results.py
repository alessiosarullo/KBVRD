import argparse
import os
import pickle
import sys
from collections import Counter

import cv2
import numpy as np

from analysis.utils import vis_one_image
from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit, Splits
from lib.dataset.utils import Minibatch
from lib.models.utils import Prediction
from lib.stats.evaluator import Evaluator


def _setup_and_load():
    cfg.parse_args()
    with open(cfg.program.result_file, 'rb') as f:
        results = pickle.load(f)
    cfg.load()
    cfg.program.load_precomputed_feats = False
    hds = HicoDetInstanceSplit.get_split(split=Splits.TEST)
    return results, hds


def vrd_style_eval_count():
    results, hds = _setup_and_load()
    stats = Evaluator.evaluate_predictions(hds, results)
    stats.print_metrics()

    gt_hois = hds.hois
    gt_hoi_hist = Counter(gt_hois[:, 1])
    num_gt_hois = sum(gt_hoi_hist.values())
    assert num_gt_hois == gt_hois.shape[0]
    num_pred_hois = stats.num_hoi_predictions_per_class
    print('  GT HOIs: [%s]' % ', '.join(['%s (%3.0f%%)' % (c.replace('_', ' ').strip(), 100 * gt_hoi_hist[i] / num_gt_hois)
                                         for i, c in enumerate(hds.predicates)]))
    print('Pred HOIs: [%s]' % ', '.join(['%s (%3.0f%%)' % (c.replace('_', ' ').strip(), 100 * num_pred_hois[i] / np.sum(num_pred_hois))
                                         for i, c in enumerate(hds.predicates)]))

    obj_class_hist = Counter(hds.obj_labels)
    num_gt_obj_classes = sum(obj_class_hist.values())
    assert num_gt_obj_classes == hds.obj_labels.shape[0]
    num_pred_objs = stats.num_obj_predictions_per_class
    print('  GT objects: [%s]' % ', '.join(['%s (%3.0f%%)' % (c.replace('_', ' ').strip(), 100 * obj_class_hist[i] / num_gt_obj_classes)
                                            for i, c in enumerate(hds.objects)]))
    print('Pred objects: [%s]' % ', '.join(['%s (%3.0f%%)' % (c.replace('_', ' ').strip(), 100 * num_pred_objs[i] / np.sum(num_pred_objs))
                                            for i, c in enumerate(hds.objects)]))


def count():
    results, hds = _setup_and_load()
    stats = Evaluator.evaluate_predictions(hds, results)
    stats.print_metrics()

    gt_hois = hds.hois
    gt_hoi_hist = Counter(gt_hois[:, 1])
    num_gt_hois = sum(gt_hoi_hist.values())
    print('GT HOIs: [%s]' % ', '.join(['%s (%3.0f%%)' % (c.replace('_', ' ').strip(), 100 * gt_hoi_hist[i] / num_gt_hois)
                                       for i, c in enumerate(hds.predicates)]))


def vis_masks():
    results, hds = _setup_and_load()
    hdsl = hds.get_loader(batch_size=1, shuffle=False)

    output_dir = os.path.join('analysis', 'output', 'vis', *(cfg.program.output_path.split('/')[1:]))
    dataset_classes = hds.objects

    for b_idx, example in enumerate(hdsl):
        example = example  # type: Minibatch
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
        im_fn = example.other_ex_data[0]['fn']
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
        break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('func', type=str, choices=['vis', 'count'])
    namespace = parser.parse_known_args()
    func = vars(namespace[0])['func']
    sys.argv = sys.argv[:1] + namespace[1]
    print(sys.argv)
    if func == 'vis':
        vis_masks()
    elif func == 'count':
        count()
    else:
        raise ValueError('Unknown function %s.' % func)


if __name__ == '__main__':
    main()
