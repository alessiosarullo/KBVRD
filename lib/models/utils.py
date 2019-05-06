import numpy as np


class Prediction:
    def __init__(self, obj_im_inds, obj_boxes, obj_scores, ho_img_inds, ho_pairs, action_scores=None, hoi_scores=None):
        self.obj_im_inds = obj_im_inds  # type: np.ndarray
        self.obj_boxes = obj_boxes  # type: np.ndarray
        self.obj_scores = obj_scores  # type: np.ndarray
        self.ho_img_inds = ho_img_inds  # type: np.ndarray
        self.ho_pairs = ho_pairs  # type: np.ndarray
        self.action_score_distributions = action_scores  # type: np.ndarray
        self.hoi_scores = hoi_scores  # type: np.ndarray

    @classmethod
    def from_dict(cls, prediction_dict):
        p = Prediction(obj_im_inds=None, obj_boxes=None, obj_scores=None, ho_img_inds=None, ho_pairs=None, action_scores=None, hoi_scores=None)
        # if 'hoi_img_inds' in prediction_dict.keys():  # FIXME legacy, remove
        #     prediction_dict['ho_img_inds'] = prediction_dict['hoi_img_inds']
        #     del prediction_dict['hoi_img_inds']
        p.__dict__.update(prediction_dict)
        return p
