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
            if 'action_score_distribution' in prediction_dict.keys():  # FIXME legacy, remove
                prediction_dict['action_scores'] = prediction_dict['action_score_distribution']
                del prediction_dict['action_score_distribution']
            self.__dict__.update(prediction_dict)

    def from_dict(self, prediction_dict):
        return self


class VisualOutput:
    def __init__(self):
        # All Torch tensors except `ho_infos`

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

    def get_hoi_labels(self, dataset):
        interactions = dataset.hicodet.interactions
        hoi_obj_labels = self.box_labels[self.ho_infos[:, 2]].unsqueeze(dim=1)
        obj_labels_1hot = self.action_labels.new_zeros((hoi_obj_labels.shape[0], dataset.num_object_classes)).scatter_(1, hoi_obj_labels, 1.)
        hoi_labels = obj_labels_1hot[:, interactions[:, 1]] * self.action_labels[:, interactions[:, 0]]
        assert hoi_labels.shape[0] == self.action_labels.shape[0] and hoi_labels.shape[1] == dataset.hicodet.num_interactions
        return  hoi_labels

    def filter_boxes(self, valid_box_mask=None):
        assert self.boxes_ext is not None

        if valid_box_mask is None:
            assert self.box_labels is not None
            valid_box_mask = (self.box_labels >= 0)

        discarded_boxes_ext = self.boxes_ext[~valid_box_mask, :]
        discarded_box_feats = self.box_feats[~valid_box_mask, :]
        discarded_masks = self.masks[~valid_box_mask, :]

        if not valid_box_mask.any():
            self.boxes_ext = None
            self.box_feats = None
            self.masks = None
            self.box_labels = None
        else:
            self.boxes_ext = self.boxes_ext[valid_box_mask, :]
            self.box_feats = self.box_feats[valid_box_mask, :]
            self.masks = self.masks[valid_box_mask, :]
            if self.box_labels is not None:
                self.box_labels = self.box_labels[valid_box_mask]

        if self.ho_infos is not None:
            fg_box_mask_np = valid_box_mask.detach().cpu().numpy().astype(bool)
            fg_box_inds_index = np.full(fg_box_mask_np.shape[0], fill_value=-1, dtype=np.int)
            fg_box_inds_index[fg_box_mask_np] = np.arange(fg_box_mask_np.sum())

            ho_infos = self.ho_infos.copy()
            ho_infos[:, 1] = fg_box_inds_index[ho_infos[:, 1]]
            ho_infos[:, 2] = fg_box_inds_index[ho_infos[:, 2]]

            valid_hoi_mask = np.all(ho_infos >= 0, axis=1)

            if not np.any(valid_hoi_mask):
                self.ho_infos = None
                self.hoi_union_boxes = None
                self.hoi_union_boxes_feats = None
                self.action_labels = None
            else:
                self.ho_infos = ho_infos[valid_hoi_mask, :]
                self.hoi_union_boxes = self.hoi_union_boxes[valid_hoi_mask, :]

                valid_hoi_mask = (torch.from_numpy(valid_hoi_mask.astype(np.uint8)) > 0)
                self.hoi_union_boxes_feats = self.hoi_union_boxes_feats[valid_hoi_mask, :]

                if self.action_labels is not None:
                    self.action_labels = self.action_labels[valid_hoi_mask, :]

        return discarded_boxes_ext, discarded_box_feats, discarded_masks