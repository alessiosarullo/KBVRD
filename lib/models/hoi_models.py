import math

import numpy as np
import torch
import torch.nn as nn

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.word_embeddings import WordEmbeddings
from lib.knowledge_extractors.imsitu_knowledge_extractor import ImSituKnowledgeExtractor
from lib.models.abstract_model import AbstractHOIBranch
from lib.models.context_modules import SpatialContext, ObjectContext
from lib.models.generic_model import GenericModel
from lib.models.nmotifs.freq import FrequencyBias


# from .highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM


class BaseModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'base'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        # FIXME? Since batches are fairly small due to memory constraint, BN might not be suitable. Maybe switch to GN?
        self.use_bn = False
        self.imsitu_prior_emb_dim = 256
        self.imsitu_branch_final_emb_dim = 512
        super().__init__(dataset, **kwargs)
        self.use_int_freq = cfg.model.use_int_freq
        self.use_ext_imsitu = cfg.model.use_imsitu

        self.spatial_context_branch = SpatialContext(input_dim=2 * (self.visual_module.mask_resolution ** 2))
        self.obj_branch = ObjectContext(input_dim=self.visual_module.vis_feat_dim +
                                                  self.dataset.num_object_classes +
                                                  self.spatial_context_branch.output_dim)
        self.hoi_branch = BaseHOIBranch(self.visual_module.vis_feat_dim,
                                        self.spatial_context_branch.spatial_emb_dim,
                                        self.obj_branch.output_ctx_dim,
                                        self.spatial_context_branch.output_dim,
                                        self.dataset.objects,
                                        **kwargs)

        self.obj_output_fc = nn.Linear(self.obj_branch.output_feat_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, self.dataset.num_predicates)

        if self.use_ext_imsitu:
            self.imsitu_prior_obj_attention_fc = nn.Sequential(nn.Linear(self.obj_branch.output_feat_dim, self.dataset.num_object_classes),
                                                               nn.Softmax(dim=1))
            imsitu_ke = ImSituKnowledgeExtractor()
            imsitu_prior = imsitu_ke.extract_prior_matrix(self.dataset)
            imsitu_prior = np.log(imsitu_prior / np.maximum(1, np.sum(imsitu_prior, axis=1, keepdims=True)) + 1e-3)  # FIXME magic constant
            self.imsitu_prior = torch.nn.Embedding.from_pretrained(torch.from_numpy(imsitu_prior).float(), freeze=False)

        if self.use_int_freq:
            self.freq_bias = FrequencyBias(dataset=dataset)

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

        if self.use_ext_imsitu:
            imsitu_prior_per_obj = torch.mm(self.imsitu_prior_obj_attention_fc(objs_embs), self.imsitu_prior.weight)
            hoi_obj_imsitu_prior = imsitu_prior_per_obj[hoi_infos[:, 2], :]
            hoi_logits += hoi_obj_imsitu_prior

        if self.use_int_freq:
            hoi_infos = torch.tensor(hoi_infos, device=masks.device)
            obj_classes = torch.argmax(boxes_ext[:, 5:], dim=1)  # FIXME in nmotifs they use actual labels
            hoi_logits = hoi_logits + self.freq_bias.index_with_labels(obj_classes[hoi_infos[:, 2]])

        return obj_logits, hoi_logits


class BaseHOIBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, spatial_emb_dim, obj_ctx_dim, spatial_ctx_dim, objects_class_names, **kwargs):
        def _vis_fc_layer():
            return nn.Sequential(*([nn.Linear(visual_feats_dim, self.hoi_visual_hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(self.hoi_visual_hidden_dim, self.hoi_visual_hidden_dim)]
                                   +
                                   ([nn.BatchNorm1d(self.hoi_visual_hidden_dim)] if self.use_bn else [])
                                   ))

        self.hoi_visual_hidden_dim = 1024
        self.hoi_emb_dim = 1024
        self.word_emb_dim = 200
        self.use_bn = False
        super().__init__(**kwargs)

        # TODO? Maybe use BN with no momentum instead of setting the standard deviation manually?
        self.input_vis_feats_fc = nn.Linear(visual_feats_dim, visual_feats_dim)
        torch.nn.init.normal_(self.input_vis_feats_fc.weight, mean=0, std=math.sqrt(1.0 / visual_feats_dim))

        self.input_sp_feats_fc = nn.Linear(spatial_emb_dim, spatial_emb_dim)
        torch.nn.init.normal_(self.input_sp_feats_fc.weight, mean=0, std=math.sqrt(1.0 / spatial_emb_dim))

        self.input_sp_ctx_fc = nn.Linear(spatial_ctx_dim, spatial_ctx_dim)
        torch.nn.init.normal_(self.input_sp_ctx_fc.weight, mean=0, std=10 * math.sqrt(1.0 / spatial_ctx_dim))

        self.input_obj_ctx_fc = nn.Linear(obj_ctx_dim, obj_ctx_dim)
        torch.nn.init.normal_(self.input_obj_ctx_fc.weight, mean=0, std=10 * math.sqrt(1.0 / obj_ctx_dim))

        word_embeddings = WordEmbeddings(source='glove', dim=self.word_emb_dim).get_embeddings(objects_class_names)  # FIXME magic constant
        num_object_classes = len(objects_class_names)
        self.word_embs = torch.nn.Embedding.from_pretrained(torch.from_numpy(word_embeddings), freeze=True)
        self.word_embs_attention = nn.Sequential(nn.Linear(num_object_classes, num_object_classes),
                                                 nn.Softmax(dim=1))

        self.rel_sub_fc = _vis_fc_layer()
        self.rel_obj_fc = _vis_fc_layer()
        self.rel_union_fc = _vis_fc_layer()

        hoi_spatial_input_dim = self.hoi_visual_hidden_dim + spatial_emb_dim + spatial_ctx_dim
        self.rel_spatial_emb_fc = nn.Sequential(*(
                [nn.Linear(hoi_spatial_input_dim, self.hoi_emb_dim),
                 nn.ReLU(inplace=True),
                 nn.Linear(self.hoi_emb_dim, self.hoi_emb_dim)]
                +
                ([nn.BatchNorm1d(self.hoi_emb_dim)] if self.use_bn else [])
        ))

        hoi_final_input_dim = self.hoi_emb_dim + obj_ctx_dim + self.word_emb_dim
        self.rel_final_emb_fc = nn.Sequential(*(
                [nn.Linear(hoi_final_input_dim, self.hoi_emb_dim),
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

        spatial_ctx_rep = torch.cat([in_spatial_ctx[i, :].expand((hoi_im_ids == im_id).sum(), -1) for i, im_id in enumerate(unique_im_ids)], dim=0)
        hum_feats = self.rel_sub_fc(in_box_feats[sub_inds, :])
        obj_feats = self.rel_obj_fc(in_box_feats[obj_inds, :])
        union_feats = self.rel_union_fc(in_union_boxes_feats)
        rel_vis_feats = hum_feats * obj_feats * union_feats

        rel_feats1 = torch.cat([rel_vis_feats, in_spatial_rels_feats, spatial_ctx_rep], dim=1)
        rel_emb1 = self.rel_spatial_emb_fc(rel_feats1)

        wemb_attention_w = self.word_embs_attention(boxes_ext[:, 5:])
        wembs = torch.mm(wemb_attention_w, self.word_embs.weight)  # attended embeddings
        obj_wembs = wembs[obj_inds, :]  # getting the word embedding for the subject doesn't make a lot of sense, since it's always "person"
        objs_ctx_rep = torch.cat([in_obj_ctx[i, :].expand((hoi_im_ids == im_id).sum(), -1) for i, im_id in enumerate(unique_im_ids)], dim=0)
        rel_feats2 = torch.cat([rel_emb1, objs_ctx_rep, obj_wembs], dim=1)
        rel_emb_final = self.rel_final_emb_fc(rel_feats2)

        return rel_emb_final
