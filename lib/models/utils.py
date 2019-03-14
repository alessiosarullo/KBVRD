import numpy as np


class Prediction:
    def __init__(self, obj_im_inds, obj_boxes, obj_scores, hoi_img_inds, ho_pairs, hoi_scores):
        self.obj_im_inds = obj_im_inds  # type: np.ndarray
        self.obj_boxes = obj_boxes  # type: np.ndarray
        self.obj_scores = obj_scores  # type: np.ndarray
        self.hoi_img_inds = hoi_img_inds  # type: np.ndarray
        self.ho_pairs = ho_pairs  # type: np.ndarray
        self.hoi_score_distributions = hoi_scores  # type: np.ndarray

    @property
    def hoi_classes(self):
        if self.hoi_score_distributions is None:
            return None
        return self.hoi_score_distributions.argmax(axis=1)

    @property
    def obj_classes(self):
        if self.obj_scores is None:
            return None
        return self.obj_scores.argmax(axis=1)

    @classmethod
    def from_dict(cls, prediction_dict):
        p = Prediction(obj_im_inds=None, obj_boxes=None, obj_scores=None, hoi_img_inds=None, ho_pairs=None, hoi_scores=None)
        assert set(vars(p).keys()) == set(prediction_dict.keys())
        p.__dict__.update(prediction_dict)
        return p

    def is_complete(self):
        complete = True
        for v in vars(self).values():
            complete = complete and v is not None
        return complete
