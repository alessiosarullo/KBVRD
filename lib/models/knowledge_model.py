import torch
import torch.nn as nn

from lib.dataset.hicodet import HicoDetInstance
from .abstract_model import AbstractModel, AbstractHOIModule


# from .highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM


class ImSituModel(AbstractModel):
    def __init__(self, dataset: HicoDetInstance, **kwargs):
        super().__init__(dataset, **kwargs)

    def _get_hoi_branch(self):
        return ImSituHOIModule(self.mask_rcnn_vis_feat_dim, self.obj_branch.output_ctx_dim, self.spatial_context_branch.output_dim)


class ImSituHOIModule(AbstractHOIModule):
    def __init__(self, vis_feats_dim, obj_ctx_dim, spatial_ctx_dim, **kwargs):
        super().__init__()

        # FIXME params
        self.use_bn = False  # Since the batches are small due to memory constraint, BN is not suitable. TODO Maybe switch to GN?
        self.rel_vis_hidden_dim = 1024
        self.rel_emb_dim = 1024
        self.rel_dropout = 0.1
        self.filter_rels_of_non_overlapping_boxes = False  # TODO? create config for this
        self.__dict__.update({k: v for k, v in kwargs.items() if k in self.__dict__.keys() and v is not None})

        self.rel_sub_fc = nn.Linear(vis_feats_dim, self.rel_vis_hidden_dim)
        self.rel_obj_fc = nn.Linear(vis_feats_dim, self.rel_vis_hidden_dim)
        self.rel_union_fc = nn.Linear(vis_feats_dim, self.rel_vis_hidden_dim)
        self.rel_output_emb_fc = nn.Sequential(*(
                ([nn.BatchNorm1d(self.rel_vis_hidden_dim + obj_ctx_dim + spatial_ctx_dim)] if self.use_bn else [])  # 2 = biLSTM
                +
                [nn.Linear(self.rel_vis_hidden_dim + obj_ctx_dim + spatial_ctx_dim, self.rel_emb_dim),
                 nn.ReLU(inplace=True),
                 nn.Dropout(self.rel_dropout)]
                +
                ([nn.BatchNorm1d(self.rel_emb_dim)] if self.use_bn else [])
        ))

    def _forward(self, union_boxes_feats, box_feats, spatial_ctx, obj_ctx, unique_im_ids, hoi_im_ids, sub_inds, obj_inds):
        # TODO docs
        # Every input is a Tensor
        objs_ctx_rep = torch.cat([obj_ctx[i, :].expand((hoi_im_ids == im_id).sum(), -1) for i, im_id in enumerate(unique_im_ids)], dim=0)
        spatial_ctx_rep = torch.cat([spatial_ctx[i, :].expand((hoi_im_ids == im_id).sum(), -1) for i, im_id in enumerate(unique_im_ids)], dim=0)
        subj_feats = self.rel_sub_fc(box_feats[sub_inds, :])
        obj_feats = self.rel_obj_fc(box_feats[obj_inds, :])
        union_feats = self.rel_union_fc(union_boxes_feats)
        rel_vis_feats = subj_feats * obj_feats * union_feats
        rel_feats = torch.cat([rel_vis_feats, objs_ctx_rep, spatial_ctx_rep], dim=1)
        rel_emb = self.rel_output_emb_fc(rel_feats)
        return rel_emb

    @property
    def output_dim(self):
        return self.rel_emb_dim
