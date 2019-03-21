import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import Minibatch
from lib.models.abstract_model import AbstractModel
from lib.models.visual_modules import VisualModule
from lib.models.utils import Prediction


class GenericModel(AbstractModel):
    @classmethod
    def get_cline_name(cls) -> str:
        raise NotImplementedError()

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        # FIXME? params
        self.gt_iou_thr = 0.5
        super().__init__(**kwargs)
        self.dataset = dataset
        self.visual_module = VisualModule(dataset, **kwargs)

    def get_losses(self, x, **kwargs):
        obj_output, hoi_output, box_labels, hoi_labels = self(x, inference=False, **kwargs)
        obj_loss = nn.functional.cross_entropy(obj_output, box_labels)
        hoi_loss = nn.functional.binary_cross_entropy_with_logits(hoi_output, hoi_labels) * self.dataset.num_predicates
        return {'object_loss': obj_loss, 'hoi_loss': hoi_loss}

    def forward(self, x: Minibatch, inference=True, **kwargs):
        # TODO docs

        with torch.set_grad_enabled(self.training):
            boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, hoi_infos, box_labels, hoi_labels = self.visual_module(x, inference)
            # `hoi_infos` is an R x 3 NumPy array where each column is [image ID, subject index, object index].
            # Masks are floats at this point.

            # masks = masks.round().to(dtype=torch.uint8)

            if not inference:
                assert hoi_infos is not None and box_labels is not None and hoi_labels is not None
                obj_output, hoi_output = self._forward(boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels, hoi_labels)
                return obj_output, hoi_output, box_labels, hoi_labels
            else:
                if hoi_infos is not None:
                    assert boxes_ext is not None
                    obj_output, hoi_output = self._forward(boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels, hoi_labels)
                    if cfg.program.predcls:
                        obj_prob = None  # this will be assigned later as the object label distribution
                    else:
                        obj_prob = nn.functional.softmax(obj_output, dim=1).cpu().numpy()
                    hoi_probs = torch.sigmoid(hoi_output).cpu().numpy()
                    hoi_img_inds = hoi_infos[:, 0]
                    ho_pairs = hoi_infos[:, 1:]
                else:
                    hoi_probs = ho_pairs = hoi_img_inds = None
                    obj_prob = None

                if boxes_ext is not None:
                    im_scales = x.img_infos[:, 2].cpu().numpy()
                    boxes_ext = boxes_ext.cpu().numpy()
                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    if obj_prob is None:
                        obj_prob = boxes_ext[:, 5:]  # this cannot be refined because of the lack of spatial relationships
                else:
                    obj_im_inds = obj_boxes = None
                return Prediction(obj_im_inds=obj_im_inds,
                                  obj_boxes=obj_boxes,
                                  obj_scores=obj_prob,
                                  hoi_img_inds=hoi_img_inds,
                                  ho_pairs=ho_pairs,
                                  hoi_scores=hoi_probs)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        raise NotImplementedError()
