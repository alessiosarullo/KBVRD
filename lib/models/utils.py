from typing import Dict, Type, Set

import numpy as np

from lib.models.abstract_model import AbstractModel


# noinspection PyUnresolvedReferences
def get_all_models_by_name() -> Dict[str, Type[AbstractModel]]:
    # Importing is needed because otherwise subclasses are not registered. FIXME maybe?
    from lib.models.hoi_models import BaseModel, KModel
    from lib.models.nmotifs.hoi_nmotifs import HOINMotifs, HOINMotifsHybrid

    def get_all_subclasses(cls):
        return set(cls.__subclasses__()).union([s for c in cls.__subclasses__() for s in get_all_subclasses(c)])

    all_model_classes = get_all_subclasses(AbstractModel)  # type: Set[Type[AbstractModel]]
    all_model_classes_dict = {}
    for model in all_model_classes:
        try:
            all_model_classes_dict[model.get_cline_name()] = model
        except NotImplementedError:
            pass
    return all_model_classes_dict


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
