import torch
import torch.nn as nn
import math

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from .abstract_model import AbstractModel, AbstractHOIModule


# from .highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM


class BaseModel(AbstractModel):
    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _get_hoi_branch(self):
        return SimpleHOIModule(self.mask_rcnn_vis_feat_dim,
                               self.spatial_context_branch.spatial_emb_dim,
                               self.obj_branch.output_ctx_dim,
                               self.spatial_context_branch.output_dim)


class SimpleHOIModule(AbstractHOIModule):
    def __init__(self, visual_feats_dim, spatial_emb_dim, obj_ctx_dim, spatial_ctx_dim):
        super().__init__()

        def _vis_fc_layer():
            return nn.Sequential(*([nn.Linear(visual_feats_dim, self.hoi_visual_hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.hoi_visual_hidden_dim, self.hoi_visual_hidden_dim)]
                                   +
                                   ([nn.BatchNorm1d(self.hoi_visual_hidden_dim)] if self.use_bn else [])
                                   ))

        # FIXME params
        self.use_bn = cfg.model.bn  # Since batches are fairly small due to memory constraint, BN might not be suitable. Maybe switch to GN?
        self.hoi_visual_hidden_dim = 1024
        self.hoi_emb_dim = 1024

        # TODO? Maybe use BN with no momentum instead of setting the standard deviation manually?
        self.input_vis_feats_fc = nn.Linear(visual_feats_dim, visual_feats_dim)
        torch.nn.init.normal_(self.input_vis_feats_fc.weight, mean=0, std=math.sqrt(1.0 / visual_feats_dim))

        self.input_sp_feats_fc = nn.Linear(spatial_emb_dim, spatial_emb_dim)
        torch.nn.init.normal_(self.input_sp_feats_fc.weight, mean=0, std=math.sqrt(1.0 / spatial_emb_dim))

        self.input_sp_ctx_fc = nn.Linear(spatial_ctx_dim, spatial_ctx_dim)
        torch.nn.init.normal_(self.input_sp_ctx_fc.weight, mean=0, std=10 * math.sqrt(1.0 / spatial_ctx_dim))

        self.input_obj_ctx_fc = nn.Linear(obj_ctx_dim, obj_ctx_dim)
        torch.nn.init.normal_(self.input_obj_ctx_fc.weight, mean=0, std=10 * math.sqrt(1.0 / obj_ctx_dim))

        self.rel_sub_fc = _vis_fc_layer()
        self.rel_obj_fc = _vis_fc_layer()
        self.rel_union_fc = _vis_fc_layer()

        hoi_input_feat_dim = self.hoi_visual_hidden_dim + spatial_emb_dim + spatial_ctx_dim + obj_ctx_dim
        self.rel_output_emb_fc = nn.Sequential(*(
            [nn.Linear(hoi_input_feat_dim, self.hoi_emb_dim),
             nn.ReLU(inplace=True),
             nn.Linear(self.hoi_emb_dim, self.hoi_emb_dim)]
            +
            ([nn.BatchNorm1d(self.hoi_emb_dim)] if self.use_bn else [])
        ))

    def _forward(self, union_boxes_feats, spatial_rels_feats, box_feats, spatial_ctx, obj_ctx, unique_im_ids, hoi_im_ids, sub_inds, obj_inds):
        # TODO docs
        # Every input is a Tensor
        in_union_boxes_feats = self.input_vis_feats_fc(union_boxes_feats)
        in_box_feats = self.input_vis_feats_fc(box_feats)
        in_spatial_rels_feats = self.input_sp_feats_fc(spatial_rels_feats)
        in_spatial_ctx = self.input_sp_ctx_fc(spatial_ctx)
        in_obj_ctx = self.input_obj_ctx_fc(obj_ctx)

        objs_ctx_rep = torch.cat([in_obj_ctx[i, :].expand((hoi_im_ids == im_id).sum(), -1) for i, im_id in enumerate(unique_im_ids)], dim=0)
        spatial_ctx_rep = torch.cat([in_spatial_ctx[i, :].expand((hoi_im_ids == im_id).sum(), -1) for i, im_id in enumerate(unique_im_ids)], dim=0)
        subj_feats = self.rel_sub_fc(in_box_feats[sub_inds, :])
        obj_feats = self.rel_obj_fc(in_box_feats[obj_inds, :])
        union_feats = self.rel_union_fc(in_union_boxes_feats)
        rel_vis_feats = subj_feats * obj_feats * union_feats
        rel_feats = torch.cat([rel_vis_feats, in_spatial_rels_feats, spatial_ctx_rep, objs_ctx_rep], dim=1)
        rel_emb = self.rel_output_emb_fc(rel_feats)

        self.values_to_monitor['visual-boxes'] = in_box_feats
        self.values_to_monitor['visual-union_boxes_feats'] = in_union_boxes_feats
        self.values_to_monitor['spatial'] = in_spatial_rels_feats
        self.values_to_monitor['obj_ctx_rep'] = objs_ctx_rep
        self.values_to_monitor['sp_ctx_rep'] = spatial_ctx_rep
        self.values_to_monitor['concat'] = rel_feats
        self.values_to_monitor['final_emb'] = rel_emb

        return rel_emb

    @property
    def output_dim(self):
        return self.hoi_emb_dim
