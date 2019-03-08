import math

import torch
import torch.nn as nn

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.models.abstract_model import AbstractHOIBranch
from lib.models.context import SpatialContext, ObjectContext
from lib.models.generic_model import GenericModel
from lib.word_vectors import obj_edge_vectors


# from .highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM


class BaseModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'base'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)

        self.spatial_context_branch = SpatialContext(input_dim=2 * (self.mask_rcnn.mask_resolution ** 2))
        self.obj_branch = ObjectContext(input_dim=self.mask_rcnn_vis_feat_dim +
                                                  self.dataset.num_object_classes +
                                                  self.spatial_context_branch.output_dim)
        self.hoi_branch = BaseHOIBranch(self.mask_rcnn_vis_feat_dim,
                                        self.spatial_context_branch.spatial_emb_dim,
                                        self.obj_branch.output_ctx_dim,
                                        self.spatial_context_branch.output_dim,
                                        self.dataset.num_object_classes)

        self.obj_output_fc = nn.Linear(self.obj_branch.output_feat_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, self.dataset.num_predicates)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        # TODO docs

        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        spatial_ctx, spatial_rels_feats = self.spatial_context_branch(masks, im_ids, hoi_infos)
        obj_ctx, objs_embs = self.obj_branch(boxes_ext, box_feats, spatial_ctx, im_ids, box_im_ids)
        hoi_embs = self.hoi_branch(union_boxes_feats, spatial_rels_feats, box_feats, spatial_ctx, obj_ctx, boxes_ext, im_ids, hoi_infos)

        obj_logits = self.obj_output_fc(objs_embs)
        hoi_logits = self.hoi_output_fc(hoi_embs)
        return obj_logits, hoi_logits


class BaseHOIBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, spatial_emb_dim, obj_ctx_dim, spatial_ctx_dim, num_obj_classes):
        def _vis_fc_layer():
            return nn.Sequential(*([nn.Linear(visual_feats_dim, self.hoi_visual_hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.hoi_visual_hidden_dim, self.hoi_visual_hidden_dim)]
                                   +
                                   ([nn.BatchNorm1d(self.hoi_visual_hidden_dim)] if self.use_bn else [])
                                   ))

        # FIXME? params
        self.use_bn = cfg.model.bn  # Since batches are fairly small due to memory constraint, BN might not be suitable. Maybe switch to GN?
        self.hoi_visual_hidden_dim = 1024
        self.hoi_emb_dim = 1024
        self.word_emb_dim = 200
        super().__init__()


        # TODO? Maybe use BN with no momentum instead of setting the standard deviation manually?
        self.input_vis_feats_fc = nn.Linear(visual_feats_dim, visual_feats_dim)
        torch.nn.init.normal_(self.input_vis_feats_fc.weight, mean=0, std=math.sqrt(1.0 / visual_feats_dim))

        self.input_sp_feats_fc = nn.Linear(spatial_emb_dim, spatial_emb_dim)
        torch.nn.init.normal_(self.input_sp_feats_fc.weight, mean=0, std=math.sqrt(1.0 / spatial_emb_dim))

        self.input_sp_ctx_fc = nn.Linear(spatial_ctx_dim, spatial_ctx_dim)
        torch.nn.init.normal_(self.input_sp_ctx_fc.weight, mean=0, std=10 * math.sqrt(1.0 / spatial_ctx_dim))

        self.input_obj_ctx_fc = nn.Linear(obj_ctx_dim, obj_ctx_dim)
        torch.nn.init.normal_(self.input_obj_ctx_fc.weight, mean=0, std=10 * math.sqrt(1.0 / obj_ctx_dim))

        self.word_embs = obj_edge_vectors(self.dataset.objects, wv_dim=self.word_emb_dim).detach()
        self.word_embs_attention_fc = nn.Linear(num_obj_classes, num_obj_classes)

        self.rel_sub_fc = _vis_fc_layer()
        self.rel_obj_fc = _vis_fc_layer()
        self.rel_union_fc = _vis_fc_layer()

        hoi_input_feat_dim = self.hoi_visual_hidden_dim + spatial_emb_dim + spatial_ctx_dim + obj_ctx_dim + self.word_emb_dim
        self.rel_output_emb_fc = nn.Sequential(*(
                [nn.Linear(hoi_input_feat_dim, self.hoi_emb_dim),
                 nn.ReLU(inplace=True),
                 nn.Linear(self.hoi_emb_dim, self.hoi_emb_dim)]
                +
                ([nn.BatchNorm1d(self.hoi_emb_dim)] if self.use_bn else [])
        ))

    @property
    def output_dim(self):
        return self.hoi_emb_dim

    def _forward(self, union_boxes_feats, spatial_rels_feats, box_feats, spatial_ctx, obj_ctx, boxes_ext, unique_im_ids, hoi_infos):
        # TODO docs
        # Every input is a Tensor
        hoi_im_ids = hoi_infos[:, 0]
        sub_inds = hoi_infos[:, 1]
        obj_inds = hoi_infos[:, 2]

        in_union_boxes_feats = self.input_vis_feats_fc(union_boxes_feats)
        in_box_feats = self.input_vis_feats_fc(box_feats)
        in_spatial_rels_feats = self.input_sp_feats_fc(spatial_rels_feats)
        in_spatial_ctx = self.input_sp_ctx_fc(spatial_ctx)
        in_obj_ctx = self.input_obj_ctx_fc(obj_ctx)

        wemb_attention_w = torch.nn.functional.softmax(self.word_embs_attention_fc(boxes_ext[:, 5:]))
        wembs = torch.mm(wemb_attention_w, self.word_embs)  # attended embeddings
        obj_wembs = wembs[obj_inds, :]  # getting the word embedding for the subject doesn't make a lot of sense, since it's always "person"

        objs_ctx_rep = torch.cat([in_obj_ctx[i, :].expand((hoi_im_ids == im_id).sum(), -1) for i, im_id in enumerate(unique_im_ids)], dim=0)
        spatial_ctx_rep = torch.cat([in_spatial_ctx[i, :].expand((hoi_im_ids == im_id).sum(), -1) for i, im_id in enumerate(unique_im_ids)], dim=0)
        hum_feats = self.rel_sub_fc(in_box_feats[sub_inds, :])
        obj_feats = self.rel_obj_fc(in_box_feats[obj_inds, :])
        union_feats = self.rel_union_fc(in_union_boxes_feats)
        rel_vis_feats = hum_feats * obj_feats * union_feats
        rel_feats = torch.cat([rel_vis_feats, in_spatial_rels_feats, spatial_ctx_rep, objs_ctx_rep, obj_wembs], dim=1)
        rel_emb = self.rel_output_emb_fc(rel_feats)

        self.values_to_monitor['visual-boxes'] = in_box_feats
        self.values_to_monitor['visual-union_boxes_feats'] = in_union_boxes_feats
        self.values_to_monitor['spatial'] = in_spatial_rels_feats
        self.values_to_monitor['obj_ctx_rep'] = objs_ctx_rep
        self.values_to_monitor['sp_ctx_rep'] = spatial_ctx_rep
        self.values_to_monitor['concat'] = rel_feats
        self.values_to_monitor['final_emb'] = rel_emb

        return rel_emb


class SimpleModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'simple'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.hoi_branch = SimpleHOIBranch(self.mask_rcnn_vis_feat_dim)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, self.dataset.num_predicates)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        # TODO docs
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        sub_inds = hoi_infos[:, 1]
        obj_inds = hoi_infos[:, 2]
        hoi_embs = self.hoi_branch(union_boxes_feats, box_feats, sub_inds, obj_inds)

        obj_logits = boxes_ext[:, 5:]
        hoi_logits = self.hoi_output_fc(hoi_embs)
        return obj_logits, hoi_logits


class SimpleHOIBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim):
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

        self.rel_sub_fc = _vis_fc_layer()
        self.rel_obj_fc = _vis_fc_layer()
        self.rel_union_fc = _vis_fc_layer()

        hoi_input_feat_dim = self.hoi_visual_hidden_dim
        self.rel_output_emb_fc = nn.Sequential(*(
                [nn.Linear(hoi_input_feat_dim, self.hoi_emb_dim),
                 nn.ReLU(inplace=True),
                 nn.Linear(self.hoi_emb_dim, self.hoi_emb_dim)]
                +
                ([nn.BatchNorm1d(self.hoi_emb_dim)] if self.use_bn else [])
        ))

    @property
    def output_dim(self):
        return self.hoi_emb_dim

    def _forward(self, union_boxes_feats, box_feats, sub_inds, obj_inds):
        # TODO docs
        # Every input is a Tensor
        in_union_boxes_feats = self.input_vis_feats_fc(union_boxes_feats)
        in_box_feats = self.input_vis_feats_fc(box_feats)

        subj_feats = self.rel_sub_fc(in_box_feats[sub_inds, :])
        obj_feats = self.rel_obj_fc(in_box_feats[obj_inds, :])
        union_feats = self.rel_union_fc(in_union_boxes_feats)
        rel_feats = subj_feats * obj_feats * union_feats
        rel_emb = self.rel_output_emb_fc(rel_feats)

        self.values_to_monitor['visual-boxes'] = in_box_feats
        self.values_to_monitor['visual-union_boxes_feats'] = in_union_boxes_feats
        self.values_to_monitor['concat'] = rel_feats
        self.values_to_monitor['final_emb'] = rel_emb

        return rel_emb
