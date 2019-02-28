import os
import pickle

import argparse
import cv2
import numpy as np

from config import Configs as cfg
from lib.containers import Minibatch
from lib.containers import Prediction
from lib.dataset.hicodet import HicoDetInstance, Splits
from analysis.utils import vis_one_image


def vis_masks(rescale=False):
    cfg.parse_args()
    with open(cfg.program.result_file_format % 'sgdet', 'rb') as f:
        results = pickle.load(f)
    cfg.load()
    output_dir = os.path.join('analysis', 'output', 'vis', *(cfg.program.save_dir.split('/')[1:]))

    hds = HicoDetInstance(Splits.TEST)
    hdsl = hds.get_loader(batch_size=1, shuffle=False)
    dataset_classes = hds.objects

    for b_idx, example in enumerate(hdsl):
        example = example  # type: Minibatch
        prediction_dict = results[b_idx]
        prediction = Prediction(**prediction_dict)  # type: Prediction

        boxes = prediction.obj_boxes
        if rescale:
            boxes /= example.img_infos[:, 2][prediction.obj_im_inds, None]
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
    parser.add_argument('--rescale', action='store_true')
    args = vars(parser.parse_known_args()[0])
    vis_masks(**args)


if __name__ == '__main__':
    main()
