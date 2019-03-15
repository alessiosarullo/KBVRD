import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel

from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.models.generic_model import GenericModel
from lib.models.abstract_model import AbstractHOIBranch
from lib.models.context_modules import SpatialContext, ObjectContext
from lib.models.nmotifs.freq import FrequencyBias
from lib.models.nmotifs.lincontext import LinearizedContext
from lib.knowledge_extractors.imsitu_knowledge_extractor import ImSituKnowledgeExtractor


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


class NMotifsHybrid(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'nmotifs-h'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.spatial_context_branch = SpatialContext(input_dim=2 * (self.visual_module.mask_resolution ** 2))
        self.obj_branch = ObjectContext(input_dim=self.visual_module.vis_feat_dim +
                                                  self.dataset.num_object_classes +
                                                  self.spatial_context_branch.output_dim)
        self.obj_output_fc = nn.Linear(self.obj_branch.output_feat_dim, self.dataset.num_object_classes)

        self.hoi_branch = NMotifsHOIBranch(self.dataset, self.visual_module.vis_feat_dim)

        # If not using NeuralMotifs's object logits no gradient flows back into the decoder RNN, so I'll make it explicit here.
        for name, param in self.named_parameters():
            if 'decoder_rnn' in name:
                param.requires_grad = False

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        hoi_im_ids = hoi_infos[:, 0]
        sub_inds = hoi_infos[:, 1]
        obj_inds = hoi_infos[:, 2]
        im_ids = torch.unique(hoi_im_ids, sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        spatial_ctx, spatial_rels_feats = self.spatial_context_branch(masks, im_ids, hoi_im_ids, sub_inds, obj_inds)
        obj_ctx, objs_embs = self.obj_branch(boxes_ext, box_feats, spatial_ctx, im_ids, box_im_ids)
        obj_logits = self.obj_output_fc(objs_embs)

        _, hoi_logits = self.hoi_branch(boxes_ext, box_feats, hoi_infos, union_boxes_feats, box_labels)
        return obj_logits, hoi_logits


class NMotifsHOIBranch(AbstractHOIBranch):
    def __init__(self, dataset: HicoDetInstanceSplit, visual_feats_dim):
        super().__init__()

        self.context = LinearizedContext(classes=dataset.objects, visual_feat_dim=visual_feats_dim)

        # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1 (half contribution comes from LSTM, half from embedding).
        # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
        self.post_lstm = nn.Linear(self.context.edge_ctx_dim, visual_feats_dim * 2)
        torch.nn.init.normal_(self.post_lstm.weight, mean=0, std=10 * math.sqrt(1.0 / self.context.edge_ctx_dim))
        torch.nn.init.zeros_(self.post_lstm.bias)

        self.hoi_output_fc = nn.Linear(visual_feats_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        self.freq_bias = FrequencyBias(dataset=dataset)

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
        hoi_logits = hoi_logits + self.freq_bias.index_with_labels(torch.stack((obj_classes[sub_inds], obj_classes[obj_inds]), dim=1))

        return obj_logits, hoi_logits


class NMotifsKB(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'nmotifs-kb'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.hoi_branch = NMotifsKBHOIBranch(self.dataset, self.visual_module.vis_feat_dim)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        obj_logits, hoi_logits = self.hoi_branch(boxes_ext, box_feats, hoi_infos, union_boxes_feats, box_labels)
        return obj_logits, hoi_logits


class NMotifsKBHOIBranch(AbstractHOIBranch):
    def __init__(self, dataset: HicoDetInstanceSplit, visual_feats_dim):
        super().__init__()

        self.context = LinearizedContext(classes=dataset.objects, visual_feat_dim=visual_feats_dim)

        # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1 (half contribution comes from LSTM, half from embedding).
        # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
        self.post_lstm = nn.Linear(self.context.edge_ctx_dim, visual_feats_dim * 2)
        torch.nn.init.normal_(self.post_lstm.weight, mean=0, std=10 * math.sqrt(1.0 / self.context.edge_ctx_dim))
        torch.nn.init.zeros_(self.post_lstm.bias)

        self.hoi_output_fc = nn.Linear(visual_feats_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        self.freq_bias = FrequencyBias(dataset=dataset)

        # KB
        eps = 1e-3
        imsitu_prior_matrix = ImSituKnowledgeExtractor().extract_prior_matrix(dataset)
        obj_pred_prior = np.log(imsitu_prior_matrix / np.maximum(1, np.sum(imsitu_prior_matrix, axis=1, keepdims=True)) + eps)
        self.imsitu_prior = nn.Embedding.from_pretrained(torch.from_numpy(obj_pred_prior).float(), freeze=True)

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
        hoi_logits = hoi_logits + self.freq_bias.index_with_labels(torch.stack((obj_classes[sub_inds], obj_classes[obj_inds]), dim=1))

        imsitu_hoi_logits = self.imsitu_prior(obj_inds)
        hoi_logits += imsitu_hoi_logits

        return obj_logits, hoi_logits
