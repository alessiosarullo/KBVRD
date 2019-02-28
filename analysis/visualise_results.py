import os
import pickle

import cv2
import numpy as np

from config import Configs as cfg
from lib.containers import Minibatch
from lib.containers import Prediction
from lib.dataset.hicodet import HicoDetInstance, Splits
from lib.detection.wrappers import vis_utils


class DummyDataset:
    def __init__(self, classes):
        self.classes = classes


def vis_masks():
    cfg.parse_args()
    output_dir = os.path.join('analysis', 'output', 'vis')
    with open(cfg.program.result_file_format % 'sgdet', 'rb') as f:
        results = pickle.load(f)
    cfg.load()

    hds = HicoDetInstance(Splits.TEST)
    hdsl = hds.get_loader(batch_size=1, shuffle=False)
    dataset = DummyDataset(hds.objects)

    for example in hdsl:
        example = example  # type: Minibatch
        prediction = results[example.other_ex_data[0]['index']]  # type: Prediction

        boxes = prediction.obj_boxes.cpu().numpy()
        obj_scores = prediction.obj_scores.cpu().numpy()
        box_classes = np.argmax(obj_scores, axis=1)
        box_class_scores = obj_scores[np.arange(boxes.shape[0]), box_classes]
        print(boxes)
        print(box_classes)
        print(box_class_scores)

        boxes_with_scores = np.concatenate([boxes, box_class_scores[:, None]], axis=1)

        im_fn = example.other_ex_data[0]['fn']
        im = cv2.imread(os.path.join(hds.img_dir, im_fn))

        cls_boxes = [boxes_with_scores[box_classes == j, :] for j in range(len(dataset.classes))]  # background is included

        vis_utils.vis_one_image(
            im[:, :, [2, 1, 0]],  # BGR -> RGB for visualization
            os.path.splitext(im_fn)[0],
            output_dir,
            cls_boxes,
            None,
            None,
            dataset=dataset,
            box_alpha=0.3,
            show_class=True,
            thresh=0.0,  # Lower this to see all the predictions (was 0.7 in the original code)
            kp_thresh=2,
            ext='png'
        )
        break


if __name__ == '__main__':
    vis_masks()
