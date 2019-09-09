from typing import Union

import numpy as np
import torch


class Prediction:
    def __init__(self, prediction_dict=None):
        self.obj_im_inds = None  # type: Union[None, np.ndarray]
        self.obj_boxes = None  # type: Union[None, np.ndarray]
        self.obj_scores = None  # type: Union[None, np.ndarray]
        self.ho_img_inds = None  # type: Union[None, np.ndarray]
        self.ho_pairs = None  # type: Union[None, np.ndarray]
        self.action_scores = None  # type: Union[None, np.ndarray]
        self.hoi_scores = None  # type: Union[None, np.ndarray]

        if prediction_dict is not None:
            self.__dict__.update(prediction_dict)


class VisualOutput:
    def __init__(self):
        # All Torch tensors except when specified

        # Object attributes
        self.boxes_ext = None  # N x 85, each [img_id, x1, y1, x2, y2, scores]
        self.box_feats = None  # N x F, where F is the dimensionality of visual features
        # self.masks = None  # N x M x M (of floats), where M is the mask resolution

        # Human-object pair attributes
        self.ho_infos_np = None  # R x 3, each [img_id, human_ind, obj_ind]
        self._ho_infos = None
        self.hoi_union_boxes = None  # R x 4, each [x1, y1, x2, y2]
        self.hoi_union_boxes_feats = None  # R x F

        # Labels
        self.box_labels = None  # N
        self.action_labels = None  # N x #actions
        self.hoi_labels = None  # N x #interactions

    @property
    def ho_infos(self):
        if self._ho_infos is None and self.ho_infos_np is not None:
            self._ho_infos = torch.tensor(self.ho_infos_np, device=self.box_feats.device)
        return self._ho_infos
