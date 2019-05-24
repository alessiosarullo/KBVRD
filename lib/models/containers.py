from typing import Union

import numpy as np
import torch


class Prediction:
    def __init__(self):
        self.obj_im_inds = None  # type: Union[None, np.ndarray]
        self.obj_boxes = None  # type: Union[None, np.ndarray]
        self.obj_scores = None  # type: Union[None, np.ndarray]
        self.ho_img_inds = None  # type: Union[None, np.ndarray]
        self.ho_pairs = None  # type: Union[None, np.ndarray]
        self.action_scores = None  # type: Union[None, np.ndarray]
        self.hoi_scores = None  # type: Union[None, np.ndarray]

    @classmethod
    def from_dict(cls, prediction_dict):
        p = Prediction()
        p.__dict__.update(prediction_dict)
        return p


class VisualOutput:
    def __init__(self):
        # Object attributes
        self.boxes_ext = None  # N x 85, each [img_id, x1, y1, x2, y2, scores]
        self.box_feats = None  # N x F, where F is the dimensionality of visual features
        self.masks = None  # N x M x M (of floats), where M is the mask resolution

        # Human-object pair attributes
        self.ho_infos = None  # R x 3, each [img_id, human_ind, obj_ind]
        self.hoi_union_boxes = None  # R x 4, each [x1, y1, x2, y2]
        self.hoi_union_boxes_feats = None  # R x F

        # Labels
        self.box_labels = None  # N
        self.action_labels = None  # N x #actions

    def filter_boxes(self, thr=None):
        assert self.boxes_ext is not None

        if thr is None:  # filter BG boxes
            valid_box_mask = (self.box_labels >= 0)
        else:
            thr = min(thr, self.boxes_ext[:, 5:].max())
            valid_box_mask = (self.boxes_ext[:, 5:].max(dim=1)[0] >= thr)
            assert valid_box_mask.any()

        discarded_boxes = self.boxes_ext[~valid_box_mask, :]
        discarded_box_feats = self.box_feats[~valid_box_mask, :]
        discarded_masks = self.masks[~valid_box_mask, :]

        self.boxes_ext = self.boxes_ext[valid_box_mask, :]
        self.box_feats = self.box_feats[valid_box_mask, :]
        self.masks = self.masks[valid_box_mask, :]
        if self.box_labels is not None:
            self.box_labels = self.box_labels[valid_box_mask]

        if self.ho_infos is not None:
            valid_box_mask = valid_box_mask.detach().cpu().numpy().astype(bool)
            valid_box_inds_index = np.full(valid_box_mask.shape[0], fill_value=-1, dtype=np.int)
            valid_box_inds_index[valid_box_mask] = np.arange(valid_box_mask.sum())
            self.ho_infos[:, 1] = valid_box_inds_index[self.ho_infos[:, 1]]
            self.ho_infos[:, 2] = valid_box_inds_index[self.ho_infos[:, 2]]

            valid_hoi_mask = np.all(self.ho_infos >= 0, axis=1)
            if not np.any(valid_hoi_mask) and self.box_labels is not None:  # training
                valid_hoi_mask[0] = 1

            if not np.any(valid_hoi_mask):
                self.ho_infos = None
                self.hoi_union_boxes = None
                self.hoi_union_boxes_feats = None
                self.action_labels = None
            else:
                self.ho_infos = self.ho_infos[valid_hoi_mask, :]

                valid_hoi_mask = torch.from_numpy(valid_hoi_mask.astype(np.uint8))
                self.hoi_union_boxes = self.hoi_union_boxes[valid_hoi_mask, :]
                self.hoi_union_boxes_feats = self.hoi_union_boxes_feats[valid_hoi_mask, :]

                if self.action_labels is not None:
                    self.action_labels = self.action_labels[valid_hoi_mask, :]

        return discarded_boxes, discarded_box_feats, discarded_masks
