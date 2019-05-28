import numpy as np
import torch
import torch.nn as nn

from collections import Counter

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import Minibatch
from lib.models.abstract_model import AbstractModel
from lib.detection.visual_module import VisualModule
from lib.models.containers import Prediction, VisualOutput


class GenericModel(AbstractModel):
    @classmethod
    def get_cline_name(cls) -> str:
        raise NotImplementedError()

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        self.gt_iou_thr = 0.5  # before superclass' constructor's invocation because of the dict update of attributes according to keyword arguments.
        super().__init__(**kwargs)
        self.dataset = dataset
        self.visual_module = VisualModule(dataset)

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
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos is not None:
                action_output = self._forward(vis_output)
            else:
                assert inference
                action_output = None

            if not inference:
                action_labels = vis_output.action_labels
                losses = {'action_loss': nn.functional.binary_cross_entropy_with_logits(action_output, action_labels) * action_output.shape[1]}
                return losses
            else:
                prediction = Prediction()

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2].cpu().numpy()

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes
                    prediction.obj_scores = boxes_ext[:, 5:]

                    if vis_output.ho_infos is not None:
                        assert action_output is not None

                        prediction.ho_img_inds = vis_output.ho_infos[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos[:, 1:]
                        prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()

                return prediction

    def _forward(self, vis_output: VisualOutput):
        raise NotImplementedError()
