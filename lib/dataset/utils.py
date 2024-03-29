from enum import Enum

import numpy as np
import torch

from config import cfg

COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


class Example:
    def __init__(self, idx_in_split, img_id, filename, split):
        self.index = idx_in_split
        self.id = img_id
        self.filename = filename
        self.split = split

        self.image = None
        self.gt_boxes = None
        self.gt_obj_classes = None
        self.gt_hois = None


class Splits(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


def get_hico_to_coco_mapping(hico_objects, split_objects=None):
    if split_objects is None:
        split_objects = hico_objects
    coco_obj_to_idx = {('hair dryer' if c == 'hair drier' else c).replace(' ', '_'): i for i, c in enumerate(COCO_CLASSES)}
    assert set(coco_obj_to_idx.keys()) - {'__background__'} == set(hico_objects)
    mapping = np.array([coco_obj_to_idx[obj] for obj in split_objects], dtype=np.int)
    return mapping


def interactions_to_mat(hois, hico, np2np=False):
    # Default is Torch to Torch
    if not np2np:
        hois_np = hois.detach().cpu().numpy()
    else:
        hois_np = hois
    all_hois = np.stack(np.where(hois_np > 0), axis=1)
    all_interactions = np.concatenate([all_hois[:, :1], hico.interactions[all_hois[:, 1], :]], axis=1)
    inter_mat = np.zeros((hois.shape[0], hico.num_objects, hico.num_actions))
    inter_mat[all_interactions[:, 0], all_interactions[:, 2], all_interactions[:, 1]] = 1
    if np2np:
        return inter_mat
    else:
        return torch.from_numpy(inter_mat).to(hois)
