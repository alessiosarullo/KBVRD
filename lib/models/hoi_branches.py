import math

import numpy as np
import torch
from torch import nn as nn
from torch.nn.utils.rnn import PackedSequence

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import get_counts
from lib.dataset.word_embeddings import WordEmbeddings
from lib.knowledge_extractors.conceptnet_knowledge_extractor import ConceptnetKnowledgeExtractor
from lib.knowledge_extractors.imsitu_knowledge_extractor import ImSituKnowledgeExtractor
from lib.models.abstract_model import AbstractHOIBranch


class SimpleHoiBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, obj_repr_dim, **kwargs):
        # TODO docs and FIXME comments
        self.hoi_repr_dim = 600
        super().__init__(**kwargs)

        self.union_repr_fc = nn.Linear(visual_feats_dim, self.hoi_repr_dim)
        nn.init.xavier_normal_(self.union_repr_fc.weight, gain=1.0)

        self.hoi_obj_repr_fc = nn.Linear(obj_repr_dim, self.hoi_repr_dim)
        nn.init.xavier_normal_(self.hoi_obj_repr_fc.weight, gain=1.0)

        self.hoi_subj_repr_fc = nn.Linear(obj_repr_dim, self.hoi_repr_dim)
        nn.init.xavier_normal_(self.hoi_subj_repr_fc.weight, gain=1.0)

    @property
    def output_dim(self):
        return self.hoi_repr_dim

    def _forward(self, obj_repr, union_boxes_feats, hoi_infos):
        hoi_subj_repr = self.hoi_subj_repr_fc(obj_repr[hoi_infos[:, 0], :])
        hoi_obj_repr = self.hoi_obj_repr_fc(obj_repr[hoi_infos[:, 2], :])
        union_repr = self.union_repr_fc(union_boxes_feats)
        hoi_repr = union_repr + hoi_subj_repr + hoi_obj_repr
        return hoi_repr


class HoiPriorBranch(AbstractHOIBranch):
    def __init__(self, dataset: HicoDetInstanceSplit, hoi_repr_dim, **kwargs):
        super().__init__(**kwargs)

        # Freq bias
        freqs = []
        if cfg.model.freq_bias:
            int_counts = get_counts(dataset=dataset)
            freqs.append(int_counts)
        # Possibly add here other priors

        if freqs:
            self.bias_priors = nn.ModuleList()
            for fmat in freqs:
                priors = fmat / np.maximum(1, np.sum(fmat, axis=1, keepdims=True))
                self.bias_priors.append(torch.nn.Embedding.from_pretrained(torch.from_numpy(priors).float(), freeze=not cfg.model.train_prior))

            if cfg.model.prior_att:
                self.prior_source_attention = nn.Sequential(nn.Linear(hoi_repr_dim, len(self.bias_priors)),
                                                            nn.Sigmoid())
            else:
                self.prior_source_attention = None
        else:  # no actual refinement
            self.bias_priors = None

    def _forward(self, hoi_logits, hoi_repr, obj_classes, hoi_infos):
        if self.bias_priors:
            hoi_obj_classes = obj_classes[hoi_infos[:, 2]].detach()

            priors = torch.stack([prior(hoi_obj_classes) for prior in self.bias_priors], dim=0).clamp(min=1e-3)  # FIXME magic constant

            if self.prior_source_attention is not None:
                src_att = self.prior_source_attention(hoi_repr)
                prior_contribution = (src_att.t().unsqueeze(dim=2) * priors).sum(dim=0)
                self.values_to_monitor['hoi_attention'] = src_att.detach().cpu().numpy()
            else:
                prior_contribution = priors.sum(dim=0)
            hoi_logits += prior_contribution.log()
        return hoi_logits


class HoiEmbsimBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, dataset: HicoDetInstanceSplit, **kwargs):
        # TODO docs and FIXME comments
        self.word_emb_dim = 300
        super().__init__(**kwargs)
        self.num_objects = dataset.num_object_classes
        self.num_predicates = dataset.num_predicates

        self.word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        obj_word_embs = self.word_embs.get_embeddings(dataset.objects)
        pred_word_embs = self.word_embs.get_embeddings(dataset.predicates)

        interactions = dataset.hicodet.interactions  # each is [p, o]
        hoi_embs = np.concatenate([pred_word_embs[interactions[:, 0]],
                                   obj_word_embs[interactions[:, 1]]], axis=1)
        num_interactions = interactions.shape[0]
        assert num_interactions == 600
        hoi_to_obj = np.zeros((num_interactions, dataset.num_object_classes))
        hoi_to_obj[np.arange(num_interactions), interactions[:, 1]] = 1
        hoi_to_obj /= np.maximum(1, hoi_to_obj.sum(axis=0, keepdims=True))
        hoi_to_actions = np.zeros((num_interactions, dataset.num_predicates))
        hoi_to_actions[np.arange(num_interactions), interactions[:, 0]] = 1
        hoi_to_actions /= np.maximum(1, hoi_to_actions.sum(axis=0, keepdims=True))

        self.hoi_embs = nn.Parameter(torch.from_numpy(hoi_embs.T), requires_grad=False)
        self.hoi_to_obj = nn.Parameter(torch.from_numpy(hoi_to_obj).float(), requires_grad=False)
        self.hoi_to_preds = nn.Parameter(torch.from_numpy(hoi_to_actions).float(), requires_grad=False)
        self.op_cossim = torch.nn.CosineSimilarity(dim=1)

        self.obj_vis_to_emb_fc = nn.Sequential(nn.Linear(visual_feats_dim, 2 * self.word_emb_dim),
                                               nn.ReLU(),
                                               nn.Linear(2 * self.word_emb_dim, self.word_emb_dim))
        nn.init.xavier_normal_(self.obj_vis_to_emb_fc[0].weight, gain=1.0)
        nn.init.xavier_normal_(self.obj_vis_to_emb_fc[2].weight, gain=1.0)
        self.pred_vis_to_emb_fc = nn.Sequential(nn.Linear(visual_feats_dim, 2 * self.word_emb_dim),
                                                nn.ReLU(),
                                                nn.Linear(2 * self.word_emb_dim, self.word_emb_dim))
        nn.init.xavier_normal_(self.pred_vis_to_emb_fc[0].weight, gain=1.0)
        nn.init.xavier_normal_(self.pred_vis_to_emb_fc[2].weight, gain=1.0)

        # self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)
        # torch.nn.init.xavier_normal_(self.obj_output_fc.weight, gain=1.0)
        # self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True)
        # torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

    def _forward(self, union_box_feats, box_feats, hoi_infos):
        obj_embs = self.obj_vis_to_emb_fc(box_feats)
        pred_embs = self.pred_vis_to_emb_fc(union_box_feats)

        op_embs = torch.cat([pred_embs, obj_embs[hoi_infos[:, 2], :]], dim=1)
        op_sims = self.op_cossim(op_embs.unsqueeze(dim=2), self.hoi_embs.unsqueeze(dim=0))

        hoi_obj_logits = op_sims @ self.hoi_to_obj
        action_logits = op_sims @ self.hoi_to_preds

        return action_logits, hoi_obj_logits


class KatoGCNBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, obj_repr_dim, dataset: HicoDetInstanceSplit, **kwargs):
        self.word_emb_dim = 300
        super().__init__(**kwargs)

        interactions = dataset.hicodet.interactions  # each is [p, o]
        num_interactions = interactions.shape[0]
        assert num_interactions == 600
        interactions_to_obj = np.zeros((num_interactions, dataset.num_object_classes))
        interactions_to_obj[np.arange(num_interactions), interactions[:, 1]] = 1
        interactions_to_preds = np.zeros((num_interactions, dataset.num_predicates))
        interactions_to_preds[np.arange(num_interactions), interactions[:, 0]] = 1
        self.interactions_to_obj = nn.Parameter(torch.from_numpy(interactions_to_obj).float(), requires_grad=False)
        self.interactions_to_preds = nn.Parameter(torch.from_numpy(interactions_to_preds).float(), requires_grad=False)

        adj_av = torch.from_numpy(interactions_to_preds).float()
        adj_an = torch.from_numpy(interactions_to_obj).float()
        adj_nn = torch.eye(dataset.num_object_classes).float()
        adj_vv = torch.eye(dataset.num_predicates).float()

        # Normalise. The vv and nn matrices don't need it since they are identities. I think the other ones are supposed to be normalised like
        # this, but the paper is not clear at all.
        self.adj_vv = nn.Parameter(adj_vv, requires_grad=False)
        self.adj_nn = nn.Parameter(adj_nn, requires_grad=False)
        self.adj_an = nn.Parameter((1 / torch.diag(adj_an.sum(dim=1)).sqrt()) @ adj_an @ (1 / torch.diag(adj_an.sum(dim=0)).sqrt()),
                                   requires_grad=False)
        self.adj_av = nn.Parameter((1 / torch.diag(adj_av.sum(dim=1)).sqrt()) @ adj_av @ (1 / torch.diag(adj_av.sum(dim=0)).sqrt()),
                                   requires_grad=False)

        self.word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        obj_word_embs = self.word_embs.get_embeddings(dataset.objects, retry='last')
        pred_word_embs = self.word_embs.get_embeddings(dataset.predicates, retry='first')

        self.z_n = nn.Parameter(torch.from_numpy(obj_word_embs).float(), requires_grad=False)
        self.z_v = nn.Parameter(torch.from_numpy(pred_word_embs).float(), requires_grad=False)

        gc_dims = [512, 200]
        self.gc_fc1 = nn.Sequential(nn.Linear(self.word_emb_dim, gc_dims[0]),
                                    nn.ReLU())
        self.gc_fc2 = nn.Sequential(nn.Linear(gc_dims[0], gc_dims[1]),
                                    nn.ReLU())

        # vis_dim = 512
        vis_dim = gc_dims[-1]
        self.hoi_obj_fc = nn.Linear(obj_repr_dim, vis_dim)
        nn.init.xavier_normal_(self.hoi_obj_fc.weight, gain=1.0)
        self.hoi_union_fc = nn.Linear(visual_feats_dim, vis_dim)
        nn.init.xavier_normal_(self.hoi_union_fc.weight, gain=1.0)

        self.score_mlp = nn.Sequential(nn.Linear(gc_dims[1] + vis_dim, 512),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(512, 200),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(200, 1),
                                       nn.Sigmoid())

    def _forward(self, obj_repr, union_boxes_feats, hoi_infos):
        hoi_obj_repr = self.hoi_obj_fc(obj_repr[hoi_infos[:, 2], :])
        union_repr = self.hoi_union_fc(union_boxes_feats)
        hoi_repr = union_repr + hoi_obj_repr

        # WTF. What they wrote in their paper does not seem to make sense. This is what I managed to come up with.
        z_n = self.gc_fc1(self.adj_nn @ self.z_n)
        z_v = self.gc_fc1(self.adj_vv @ self.z_v)

        z_a = self.gc_fc2(self.adj_an @ z_n) + self.gc_fc2(self.adj_av @ z_v)

        hoi_logits = nn.functional.cosine_similarity(hoi_repr.unsqueeze(dim=2), z_a.t().unsqueeze(dim=0), dim=1)
        # hoi_logits = self.score_mlp(torch.cat([hoi_repr.unsqueeze(dim=1).expand(-1, z_a.shape[0], -1),
        #                                        z_a.unsqueeze(dim=0).expand(hoi_repr.shape[0], -1, -1)],
        #                                       dim=2))
        # assert hoi_logits.shape[2] == 1
        # hoi_logits = hoi_logits.squeeze(dim=2)  # this are over the interactions

        action_logits = (hoi_logits.unsqueeze(dim=2) * self.interactions_to_preds.unsqueeze(dim=0)).max(dim=1)[0]  # over actions
        hoi_obj_logits = (hoi_logits.unsqueeze(dim=2) * self.interactions_to_obj.unsqueeze(dim=0)).max(dim=1)[0]  # over objects

        return hoi_obj_logits, action_logits


class PeyreEmbsimBranch(AbstractHOIBranch):
    def __init__(self, visual_feats_dim, dataset: HicoDetInstanceSplit, **kwargs):
        self.word_emb_dim = 300
        super().__init__(**kwargs)

        self.word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        obj_word_embs = self.word_embs.get_embeddings(dataset.objects)
        pred_word_embs = self.word_embs.get_embeddings(dataset.predicates)

        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_word_embs), requires_grad=False)
        self.pred_word_embs = nn.Parameter(torch.from_numpy(pred_word_embs), requires_grad=False)

        output_dim = 1024
        appearance_dim = 300
        spatial_dim = 400
        self.vis_to_app_mlps = nn.ModuleDict({k: nn.Linear(visual_feats_dim, appearance_dim) for k in ['sub', 'obj']})
        self.spatial_mlp = nn.Sequential(nn.Linear(8, spatial_dim),
                                         nn.Linear(spatial_dim, spatial_dim))
        self.app_to_repr_mlps = nn.ModuleDict({k: nn.Sequential(nn.Linear(appearance_dim, output_dim),
                                                                nn.ReLU(),
                                                                nn.Dropout(p=0.5),
                                                                nn.Linear(output_dim, output_dim)) for k in ['sub', 'obj']})
        self.app_to_repr_mlps['pred'] = nn.Sequential(nn.Linear(appearance_dim * 2 + spatial_dim, output_dim),
                                                      nn.ReLU(),
                                                      nn.Dropout(p=0.5),
                                                      nn.Linear(output_dim, output_dim))
        self.wemb_to_repr_mlps = nn.ModuleDict({k: nn.Sequential(nn.Linear(self.word_emb_dim, output_dim),
                                                                 nn.ReLU(),
                                                                 nn.Linear(output_dim, output_dim)) for k in ['sub', 'pred', 'obj']})

    def _forward(self, boxes_ext, box_feats, hoi_infos):
        boxes = boxes_ext[:, 1:5]
        hoi_hum_inds = hoi_infos[:, 1]
        hoi_obj_inds = hoi_infos[:, 2]
        union_boxes = torch.cat([
            torch.min(boxes[:, :2][hoi_hum_inds], boxes[:, :2][hoi_obj_inds]),
            torch.max(boxes[:, 2:][hoi_hum_inds], boxes[:, 2:][hoi_obj_inds]),
        ], dim=1)

        union_areas = (union_boxes[:, 2:] - union_boxes[:, :2]).prod(dim=1, keepdim=True)
        union_origin = union_boxes[:, :2].repeat(1, 2)
        hoi_hum_spatial_info = (boxes[hoi_hum_inds, :] - union_origin) / union_areas
        hoi_obj_spatial_info = (boxes[hoi_obj_inds, :] - union_origin) / union_areas
        spatial_info = self.spatial_mlp(torch.cat([hoi_hum_spatial_info, hoi_obj_spatial_info], dim=1))

        hoi_hum_appearance = self.vis_to_app_mlps['sub'](box_feats)[hoi_hum_inds, :]
        obj_appearance = self.vis_to_app_mlps['obj'](box_feats)
        hoi_obj_appearance = self.vis_to_app_mlps['obj'](box_feats)[hoi_obj_inds, :]

        obj_repr = self.app_to_repr_mlps['obj'](obj_appearance)
        pred_repr = self.app_to_repr_mlps['pred'](torch.cat([hoi_hum_appearance, hoi_obj_appearance, spatial_info], dim=1))

        obj_emb = self.wemb_to_repr_mlps['obj'](self.obj_word_embs)
        pred_emb = self.wemb_to_repr_mlps['pred'](self.pred_word_embs)

        obj_logits = obj_repr @ obj_emb.t()
        hoi_logits = pred_repr @ pred_emb.t()

        return obj_logits, hoi_logits
