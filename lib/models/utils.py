import numpy as np


class Prediction:
    def __init__(self, obj_im_inds, obj_boxes, obj_scores, hoi_img_inds, ho_pairs, action_scores, hoi_scores=None):
        self.obj_im_inds = obj_im_inds  # type: np.ndarray
        self.obj_boxes = obj_boxes  # type: np.ndarray
        self.obj_scores = obj_scores  # type: np.ndarray
        self.hoi_img_inds = hoi_img_inds  # type: np.ndarray
        self.ho_pairs = ho_pairs  # type: np.ndarray
        self.action_score_distributions = action_scores  # type: np.ndarray
        self.hoi_scores = hoi_scores  # type: np.ndarray

    @classmethod
    def from_dict(cls, prediction_dict):
        p = Prediction(obj_im_inds=None, obj_boxes=None, obj_scores=None, hoi_img_inds=None, ho_pairs=None, action_scores=None, hoi_scores=None)
        if 'hoi_score_distributions' in prediction_dict:  # FIXME legacy, delete
            prediction_dict['action_score_distributions'] = prediction_dict['hoi_score_distributions']
            del prediction_dict['hoi_score_distributions']
        assert set(vars(p).keys()) == set(prediction_dict.keys())
        p.__dict__.update(prediction_dict)
        assert p.action_score_distributions is None or p.hoi_scores is None
        return p
