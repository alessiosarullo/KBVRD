import math

import torch
import torch.nn as nn
import torch.nn.parallel

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.models.abstract_model import AbstractHOIBranch
from lib.models.generic_model import GenericModel
from lib.models.nmotifs.freq import FrequencyLogProbs
from lib.dataset.utils import get_counts
from lib.models.nmotifs.lincontext import LinearizedContext


class NMotifs(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'nmotifs'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.hoi_branch = NMotifsHOIBranch(self.dataset, self.visual_module.vis_feat_dim)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        obj_logits, hoi_logits = self.hoi_branch(boxes_ext, box_feats, hoi_infos, union_boxes_feats, box_labels)
        return obj_logits, hoi_logits


class NMotifsHOIBranch(AbstractHOIBranch):
    def __init__(self, dataset: HicoDetInstanceSplit, visual_feats_dim):
        super().__init__()
        self.use_bias = cfg.model.use_int_freq

        self.context = LinearizedContext(classes=dataset.objects, visual_feat_dim=visual_feats_dim)

        # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1 (half contribution comes from LSTM, half from embedding).
        # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
        self.post_lstm = nn.Linear(self.context.edge_ctx_dim, visual_feats_dim * 2)
        torch.nn.init.normal_(self.post_lstm.weight, mean=0, std=10 * math.sqrt(1.0 / self.context.edge_ctx_dim))
        torch.nn.init.zeros_(self.post_lstm.bias)

        self.hoi_output_fc = nn.Linear(visual_feats_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        if self.use_bias:
            self.freq_bias = FrequencyLogProbs(counts=get_counts(dataset=dataset))

    @property
    def output_dim(self):
        raise NotImplementedError()

    def _forward(self, boxes_ext, box_feats, hoi_infos, union_box_feats, box_labels):
        obj_logits, obj_classes, edge_ctx = self.context(box_feats, boxes_ext, box_labels)

        edge_repr = self.post_lstm(edge_ctx)
        edge_repr = edge_repr.view(edge_repr.shape[0], 2, -1)  # Split into subject and object representations

        sub_inds = hoi_infos[:, 1]
        obj_inds = hoi_infos[:, 2]

        sub_repr = edge_repr[:, 0][sub_inds]
        obj_repr = edge_repr[:, 1][obj_inds]
        hoi_repr = sub_repr * obj_repr * union_box_feats

        hoi_logits = self.hoi_output_fc(hoi_repr)
        if self.use_bias:
            hoi_logits = hoi_logits + self.freq_bias(obj_classes[obj_inds])

        return obj_logits, hoi_logits
