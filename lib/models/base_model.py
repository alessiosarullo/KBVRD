import torch
import torch.nn as nn

from config import cfg
from lib.dataset.hicodet import HicoDetInstance
from .abstract_model import AbstractModel, AbstractHOIModule


# from .highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM


class BaseModel(AbstractModel):
    def __init__(self, dataset: HicoDetInstance, **kwargs):
        super().__init__(dataset, **kwargs)

    def _get_hoi_branch(self):
        return SimpleHOIModule(self.mask_rcnn_vis_feat_dim,
                               self.spatial_context_branch.spatial_emb_dim,
                               self.obj_branch.output_ctx_dim,
                               self.spatial_context_branch.output_dim)


class SimpleHOIModule(AbstractHOIModule):
    def __init__(self, visual_feats_dim, spatial_emb_dim, obj_ctx_dim, spatial_ctx_dim):
        super().__init__()

        # FIXME params
        self.use_bn = cfg.model.hoi_bn  # Since the batches are farily small due to memory constraint, BN might not be suitable. Maybe switch to GN?
        self.hoi_visual_hidden_dim = 1024
        self.hoi_emb_dim = 1024
        self.hoi_dropout = 0.1

        self.rel_sub_fc = nn.Sequential(nn.Linear(visual_feats_dim, self.hoi_visual_hidden_dim), nn.ReLU(inplace=True))
        self.rel_obj_fc = nn.Sequential(nn.Linear(visual_feats_dim, self.hoi_visual_hidden_dim), nn.ReLU(inplace=True))
        self.rel_union_fc = nn.Sequential(nn.Linear(visual_feats_dim, self.hoi_visual_hidden_dim), nn.ReLU(inplace=True))
        hoi_input_feat_dim = self.hoi_visual_hidden_dim + spatial_emb_dim + obj_ctx_dim + spatial_ctx_dim
        self.rel_output_emb_fc = nn.Sequential(*(
            ([nn.BatchNorm1d(hoi_input_feat_dim)] if self.use_bn else [])  # 2 = biLSTM
            +
            [nn.Linear(hoi_input_feat_dim, self.hoi_emb_dim),
             nn.ReLU(inplace=True),
             nn.Dropout(self.hoi_dropout)]
        ))

        self.rel_feats_history = []

    def _forward(self, union_boxes_feats, spatial_rels_feats, box_feats, spatial_ctx, obj_ctx, unique_im_ids, hoi_im_ids, sub_inds, obj_inds):
        # TODO docs
        # Every input is a Tensor
        objs_ctx_rep = torch.cat([obj_ctx[i, :].expand((hoi_im_ids == im_id).sum(), -1) for i, im_id in enumerate(unique_im_ids)], dim=0)
        spatial_ctx_rep = torch.cat([spatial_ctx[i, :].expand((hoi_im_ids == im_id).sum(), -1) for i, im_id in enumerate(unique_im_ids)], dim=0)
        subj_feats = self.rel_sub_fc(box_feats[sub_inds, :])
        obj_feats = self.rel_obj_fc(box_feats[obj_inds, :])
        union_feats = self.rel_union_fc(union_boxes_feats)
        rel_vis_feats = subj_feats * obj_feats * union_feats
        rel_feats = torch.cat([rel_vis_feats, spatial_rels_feats, objs_ctx_rep, spatial_ctx_rep], dim=1)
        self.rel_feats_history.append(rel_feats.detach().cpu())
        rel_emb = self.rel_output_emb_fc(rel_feats)
        return rel_emb

    @property
    def output_dim(self):
        return self.hoi_emb_dim
