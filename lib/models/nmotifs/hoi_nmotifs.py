import math

import torch
import torch.nn as nn
import torch.nn.parallel

from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.models.abstract_model import AbstractModel, AbstractHOIBranch
from .freq import FrequencyBias
from .lincontext import LinearizedContext


class HOINMotifs(AbstractModel):
    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _get_hoi_branch(self):
        return HOINMotifsHOIBranch(self.dataset,
                                   self.mask_rcnn_vis_feat_dim,
                                   )


class HOINMotifsHOIBranch(AbstractHOIBranch):
    def __init__(self, dataset: HicoDetInstanceSplit, visual_feats_dim):
        super().__init__()

        self.pooling_dim = 4096  # Default used in Neural Motifs

        self.context = LinearizedContext(classes=dataset.objects, visual_feat_dim=visual_feats_dim)

        # Initialize to sqrt(1/2n) so that the outputs all have mean 0 and variance 1 (half contribution comes from LSTM, half from embedding).
        # In practice the pre-lstm stuff tends to have stdev 0.1 so I multiplied this by 10.
        self.post_lstm = nn.Linear(self.context.edge_ctx_dim, self.pooling_dim * 2)
        self.post_lstm.weight.data.normal_(0, 10.0 * math.sqrt(1.0 / self.context.edge_ctx_dim))
        self.post_lstm.bias.data.zero_()

        self.rel_compress = nn.Linear(self.pooling_dim, self.num_rels_classes, bias=True)
        self.rel_compress.weight = torch.nn.init.xavier_normal(self.rel_compress.weight, gain=1.0)

        self.freq_bias = FrequencyBias(dataset=dataset)

    def _forward(self, boxes_ext, boxes_feats, rel_inds, union_box_feats):
        obj_dists, obj_preds, edge_ctx = self.context(obj_fmaps=boxes_feats,
                                                      obj_logits=boxes_ext[:, 5:].detach(),
                                                      im_inds=boxes_ext[:, 0],
                                                      box_priors=boxes_ext[:, 1:5],
                                                      )

        edge_repr = self.post_lstm(edge_ctx)
        edge_repr = edge_repr.view(edge_repr.size(0), 2, self.pooling_dim)  # Split into subject and object representations

        sub_repr = edge_repr[:, 0][rel_inds[:, 1]]
        obj_repr = edge_repr[:, 1][rel_inds[:, 2]]
        rel_repr = sub_repr * obj_repr * union_box_feats

        rel_dists = self.rel_compress(rel_repr)
        rel_dists = rel_dists + self.freq_bias.index_with_labels(torch.stack((obj_preds[rel_inds[:, 1]], obj_preds[rel_inds[:, 2]]), dim=1))

        return rel_dists

    @property
    def output_dim(self):
        return self.hoi_emb_dim
