import numpy as np
import torch
import torch.nn as nn

from collections import Counter

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
        self.gt_iou_thr = 0.5  # before superclass' constructor's invocation because of the dict update of attributes according to keyword arguments.
        super().__init__(**kwargs)
        self.dataset = dataset
        self.visual_module = VisualModule(dataset, **kwargs)

        if cfg.model.csloss:
            prcls_hist = Counter(dataset.hoi_triplets[:, 1])
            prcls_hist = np.array([prcls_hist[i] for i in range(dataset.num_predicates)])
            num_predicate_classes = prcls_hist.size
            assert num_predicate_classes == dataset.num_predicates
            cost_matrix = np.maximum(1, np.log2(prcls_hist[None, :] / prcls_hist[:, None]))
            assert not np.any(np.isnan(cost_matrix))
            cost_matrix[np.arange(num_predicate_classes), np.arange(num_predicate_classes)] = 0

            tot_num_preds = sum(prcls_hist)
            tot_other_preds = tot_num_preds - prcls_hist
            expected_class_cost = (cost_matrix.dot(prcls_hist)) / tot_other_preds

            self.class_pos_weights = torch.nn.Parameter(torch.from_numpy(expected_class_cost).view(1, -1), requires_grad=False)
            self.class_neg_weights = torch.nn.Parameter(torch.from_numpy(cost_matrix), requires_grad=False)

    def get_losses(self, x, **kwargs):
        obj_output, action_output, hoi_output, box_labels, action_labels, hoi_labels = self(x, inference=False, **kwargs)
        losses = {}
        if obj_output is not None:
            losses['object_loss'] = nn.functional.cross_entropy(obj_output, box_labels)
        if action_output is not None:
            losses['action_loss'] = nn.functional.binary_cross_entropy_with_logits(action_output, action_labels) * action_output.shape[1]
        if hoi_output is not None:
            losses['hoi_loss'] = nn.functional.binary_cross_entropy_with_logits(hoi_output, hoi_labels) * hoi_output.shape[1]
        assert losses
        return losses

    # def focal_loss(self, logits, labels):
    #     gamma = cfg.opt.gamma
    #     s = logits
    #     t = labels
    #     m = s.clamp(min=0)  # m = max(s, 0)
    #     x = (-s.abs()).exp()
    #     z = ((s >= 0) == t.byte()).float()
    #     loss_mat = (1 + x).pow(-gamma) * (m - s * t + x * (gamma * z).exp() * (1 + x).log())
    #     loss = loss_mat.mean() * self.dataset.num_predicates
    #     return loss
    #
    # def weighted_binary_cross_entropy_with_logits(self, logits, labels, num_rels=None):
    #     if num_rels is None:
    #         num_rels = self.dataset.num_predicates
    #     if not (labels.size() == logits.size()):
    #         raise ValueError("Target size ({}) must be the same as input size ({})".format(labels.size(), logits.size()))
    #
    #     # Binary cross entropy with the addition of two sets of class weights. One set is used for positive examples the other one is used for
    #     # negative examples.
    #     m = logits.clamp(min=0)
    #     s = logits
    #     t = labels
    #     u = self.class_pos_weights
    #     v = self.class_neg_weights[labels, :]  # FIXME
    #
    #     le = ((-m).exp() + (s - m).exp()).log()
    #
    #     # Trust
    #     if u is not None and v is not None:
    #         loss = v * m - t * (v * m + u * (s - m)) + ((1 - t) * v + u * t) * le
    #     elif u is not None and v is None:
    #         loss = m - t * (m + u * (s - m)) + (1 - t + u * t) * le
    #     elif u is None and v is not None:
    #         loss = v * m - t * (v * m + s - m) + ((1 - t) * v + t) * le
    #     else:
    #         loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    #
    #     loss = loss.mean() * num_rels  # The average is computed over classes and examples, instead of only over examples. This fixes it.
    #     return loss

    def forward(self, x: Minibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, hoi_infos, box_labels, action_labels, hoi_labels = \
                self.visual_module(x, inference)
            # `hoi_infos` is an R x 3 NumPy array where each column is [image ID, subject index, object index].
            # Masks are floats at this point.

            if hoi_infos is not None:
                obj_output, action_output, hoi_output = self._forward(boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels,
                                                                      action_labels)
            else:
                obj_output = action_output = hoi_output = None

            if not inference:
                assert all([x is not None for x in (box_labels, action_labels, hoi_labels)])
                return obj_output, action_output, hoi_output, box_labels, action_labels, hoi_labels
            else:
                return self._prepare_prediction(obj_output, action_output, hoi_infos, boxes_ext, im_scales=x.img_infos[:, 2].cpu().numpy())

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        raise NotImplementedError()

    @staticmethod
    def _prepare_prediction(obj_output, action_output, hoi_output, hoi_infos, boxes_ext, im_scales):
        if hoi_infos is not None:
            assert obj_output is not None and action_output is not None and boxes_ext is not None
            if cfg.program.predcls:
                obj_prob = None  # this will be assigned later as the object label distribution
            else:
                obj_prob = nn.functional.softmax(obj_output, dim=1).cpu().numpy()
            action_probs = torch.sigmoid(action_output).cpu().numpy()
            hoi_probs = torch.sigmoid(hoi_output).cpu().numpy()
            ho_img_inds = hoi_infos[:, 0]
            ho_pairs = hoi_infos[:, 1:]
        else:
            action_probs = hoi_probs = ho_pairs = ho_img_inds = None
            obj_prob = None

        if boxes_ext is not None:
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
                          ho_img_inds=ho_img_inds,
                          ho_pairs=ho_pairs,
                          action_scores=action_probs,
                          hoi_scores=hoi_probs)
