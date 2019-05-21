import numpy as np
import torch
from torch import nn as nn

from config import cfg
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import get_counts
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.abstract_model import AbstractHOIBranch


class SimpleHoiBranch(AbstractHOIBranch):
    def __init__(self, input_feats_dim, obj_repr_dim, **kwargs):
        # TODO docs and FIXME comments
        self.hoi_repr_dim = 600
        super().__init__(**kwargs)

        self.hoi_subj_repr_fc = nn.Linear(obj_repr_dim, self.hoi_repr_dim)
        nn.init.xavier_normal_(self.hoi_subj_repr_fc.weight, gain=1.0)

        self.hoi_obj_repr_fc = nn.Linear(obj_repr_dim, self.hoi_repr_dim)
        nn.init.xavier_normal_(self.hoi_obj_repr_fc.weight, gain=1.0)

        self.union_repr_fc = nn.Linear(input_feats_dim, self.hoi_repr_dim)
        nn.init.xavier_normal_(self.union_repr_fc.weight, gain=1.0)

    @property
    def output_dim(self):
        return self.hoi_repr_dim

    def _forward(self, obj_repr, union_boxes_feats, hoi_infos):
        hoi_subj_repr = self.hoi_subj_repr_fc(obj_repr[hoi_infos[:, 1], :])
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


class ActEmbsimPredBranch(AbstractHOIBranch):
    def __init__(self, pred_input_dim, obj_input_dim, dataset: HicoDetInstanceSplit, **kwargs):
        # TODO docs and FIXME comments
        self.word_emb_dim = 300
        super().__init__(**kwargs)
        self.num_objects = dataset.num_object_classes
        self.num_predicates = dataset.num_predicates

        self.word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        pred_word_embs = self.word_embs.get_embeddings(dataset.predicates)

        self.act_embs = nn.Parameter(torch.from_numpy(pred_word_embs.T), requires_grad=False)
        self.op_cossim = torch.nn.CosineSimilarity(dim=1)

        self.sub_input_to_emb_fc = nn.Linear(obj_input_dim, self.word_emb_dim)
        nn.init.xavier_normal_(self.sub_input_to_emb_fc.weight, gain=1.0)
        self.obj_input_to_emb_fc = nn.Linear(obj_input_dim, self.word_emb_dim)
        nn.init.xavier_normal_(self.obj_input_to_emb_fc.weight, gain=1.0)
        self.pred_input_to_emb_fc = nn.Linear(pred_input_dim, self.word_emb_dim)
        nn.init.xavier_normal_(self.pred_input_to_emb_fc.weight, gain=1.0)

    def _forward(self, hoi_feats, obj_feats, hoi_infos):
        sub_repr = self.sub_input_to_emb_fc(obj_feats)
        obj_repr = self.obj_input_to_emb_fc(obj_feats)
        pred_repr = self.pred_input_to_emb_fc(hoi_feats)
        op_repr = sub_repr[hoi_infos[:, 1], :] + pred_repr + obj_repr[hoi_infos[:, 2], :]
        act_logits = self.op_cossim(op_repr.unsqueeze(dim=2), self.act_embs.unsqueeze(dim=0))
        return act_logits


class GEmbBranch(AbstractHOIBranch):
    def __init__(self, hoi_input_dim, dataset: HicoDetInstanceSplit, **kwargs):
        # TODO docs and FIXME comments
        self.word_emb_dim = 300
        self.output_emb_dim = 600
        super().__init__(**kwargs)
        self.cnet_emb_dim = 1000
        self.num_objects = dataset.num_object_classes
        self.num_predicates = dataset.num_predicates

        obj_embs, act_embs, op_sim = self.get_cnet_rotate_embs()
        self.obj_embs = nn.Parameter(torch.from_numpy(obj_embs), requires_grad=False)
        # self.act_embs = nn.Parameter(torch.from_numpy(act_embs), requires_grad=False)
        # self.cnet_op_sim = nn.Parameter(torch.from_numpy(op_sim), requires_grad=False)

        # self.word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        # obj_word_embs = self.word_embs.get_embeddings(dataset.objects)
        # pred_word_embs = self.word_embs.get_embeddings(dataset.predicates)

        self.obj_input_to_emb_fc = nn.Linear(self.cnet_emb_dim + hoi_input_dim, self.output_emb_dim)

    @property
    def output_dim(self):
        return self.output_emb_dim

    def _forward(self, hoi_repr, hoi_infos, obj_logits, box_labels=None):
        if box_labels is not None:
            obj_repr = self.obj_embs[box_labels, :]
        else:
            obj_repr = self.obj_embs[obj_logits.argmax(dim=1), :]
        new_hoi_repr = torch.cat([hoi_repr, obj_repr[hoi_infos[:, 2], :]], dim=1)
        return new_hoi_repr

    def get_cnet_rotate_embs(self):
        emb_dim = self.cnet_emb_dim
        emb_range = (24.0 + 2.0) / emb_dim  # (self.gamma.item() + self.epsilon) / hidden_dim
        PI = 3.14159265358979323846

        entity_embs = np.load('cache/rotate/entity_embedding.npy')  # FIXME path
        with open('cache/rotate/entities.dict', 'r') as f:
            ecl_idx, entity_classes = zip(*[l.strip().split('\t') for l in f.readlines()])  # the index is loaded just for assertion check.
            ecl_idx = [int(x) for x in ecl_idx]
            assert np.all(np.arange(len(ecl_idx)) == np.array(ecl_idx))
            entity_inv_index = {e: i for i, e in enumerate(entity_classes)}

        obj_embs = entity_embs[np.array([entity_inv_index[o] for o in self.dataset.objects])]
        act_embs = np.concatenate([np.zeros((1, entity_embs.shape[1])),
                                   entity_embs[np.array([entity_inv_index[p] for p in self.dataset.get_preds_for_embs()[1:]])]
                                   ], axis=0)

        rotate_rel_embs = np.load('cache/rotate/relation_embedding.npy') / (emb_range / PI)
        re_rotrel_embs = np.cos(rotate_rel_embs)
        im_rotrel_embs = np.sin(rotate_rel_embs)

        rot_op_sims = np.zeros((self.num_objects, self.num_predicates, rotate_rel_embs.shape[0]))
        re_pred, im_pred = act_embs[:, :emb_dim], act_embs[:, emb_dim:]
        re_obj, im_obj = obj_embs[:, :emb_dim][:, None, :], obj_embs[:, emb_dim:][:, None, :]
        for i in range(rotate_rel_embs.shape[0]):
            re_dist = (re_pred * re_rotrel_embs[None, i] - im_pred * im_rotrel_embs[None, i])[None, :, :] - re_obj
            im_dist = (re_pred * im_rotrel_embs[None, i] + im_pred * re_rotrel_embs[None, i])[None, :, :] - im_obj
            dist = np.linalg.norm(np.linalg.norm(np.stack([re_dist, im_dist], axis=3), ord=2, axis=3), ord=1, axis=2)
            rot_op_sims[:, :, i] = -dist
        op_sim = rot_op_sims.max(axis=2)
        op_sim = np.log((op_sim - op_sim.min()) / (op_sim.max() - op_sim.min()))
        op_sim[:, 0] = 0
        op_sim = np.log(op_sim + 1e-3)

        return obj_embs, act_embs, op_sim


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
        super().__init__(**kwargs)

        self.word_embs = WordEmbeddings(source='word2vec')
        obj_word_embs = self.word_embs.get_embeddings(dataset.objects)
        pred_word_embs = self.word_embs.get_embeddings(dataset.predicates)

        interactions = dataset.hicodet.interactions  # each is [p, o]
        person_word_emb = np.tile(obj_word_embs[dataset.human_class], reps=[interactions.shape[0], 1])
        hoi_embs = np.concatenate([person_word_emb,
                                   pred_word_embs[interactions[:, 0]],
                                   obj_word_embs[interactions[:, 1]]], axis=1)

        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_word_embs), requires_grad=False)
        self.pred_word_embs = nn.Parameter(torch.from_numpy(pred_word_embs), requires_grad=False)
        self.visual_phrases_embs = nn.Parameter(torch.from_numpy(hoi_embs), requires_grad=False)

        appearance_dim = 300
        self.vis_to_app_mlps = nn.ModuleDict({k: nn.Linear(visual_feats_dim, appearance_dim) for k in ['sub', 'obj']})

        spatial_dim = 400
        self.spatial_mlp = nn.Sequential(nn.Linear(8, spatial_dim),
                                         nn.Linear(spatial_dim, spatial_dim))

        output_dim = 1024
        self.app_to_repr_mlps = nn.ModuleDict({k: nn.Sequential(nn.Linear(appearance_dim, output_dim),
                                                                nn.ReLU(),
                                                                nn.Dropout(p=0.5),
                                                                nn.Linear(output_dim, output_dim)) for k in ['sub', 'obj']})
        self.app_to_repr_mlps['pred'] = nn.Sequential(nn.Linear(appearance_dim * 2 + spatial_dim, output_dim),
                                                      nn.ReLU(),
                                                      nn.Dropout(p=0.5),
                                                      nn.Linear(output_dim, output_dim))
        self.app_to_repr_mlps['vp'] = nn.Sequential(nn.Linear(appearance_dim * 2 + spatial_dim, output_dim),
                                                    nn.ReLU(),
                                                    nn.Dropout(p=0.5),
                                                    nn.Linear(output_dim, output_dim))

        self.wemb_to_repr_mlps = nn.ModuleDict({k: nn.Sequential(nn.Linear(self.word_embs.dim, output_dim),
                                                                 nn.ReLU(),
                                                                 nn.Linear(output_dim, output_dim)) for k in ['sub', 'pred', 'obj']})
        self.wemb_to_repr_mlps['vp'] = nn.Sequential(nn.Linear(3 * self.word_embs.dim, output_dim),
                                                     nn.ReLU(),
                                                     nn.Linear(output_dim, output_dim))

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
        spatial_info = self.spatial_mlp(torch.cat([hoi_hum_spatial_info.detach(), hoi_obj_spatial_info.detach()], dim=1))

        subj_appearance = self.vis_to_app_mlps['sub'](box_feats)
        subj_repr = self.app_to_repr_mlps['sub'](subj_appearance)
        subj_repr = torch.nn.functional.normalize(subj_repr)
        subj_emb = self.wemb_to_repr_mlps['sub'](self.obj_word_embs)
        subj_logits = subj_repr @ subj_emb.t()
        hoi_subj_logits = subj_logits[hoi_hum_inds, :]

        obj_appearance = self.vis_to_app_mlps['obj'](box_feats)
        obj_repr = self.app_to_repr_mlps['obj'](obj_appearance)
        obj_repr = torch.nn.functional.normalize(obj_repr)
        obj_emb = self.wemb_to_repr_mlps['obj'](self.obj_word_embs)
        obj_logits = obj_repr @ obj_emb.t()
        hoi_obj_logits = obj_logits[hoi_obj_inds, :]

        hoi_subj_appearance = subj_appearance[hoi_hum_inds, :]
        hoi_obj_appearance = obj_appearance[hoi_obj_inds, :]
        hoi_act_repr = self.app_to_repr_mlps['pred'](torch.cat([hoi_subj_appearance, hoi_obj_appearance, spatial_info], dim=1))
        hoi_act_repr = torch.nn.functional.normalize(hoi_act_repr)
        hoi_act_emb = self.wemb_to_repr_mlps['pred'](self.pred_word_embs)
        hoi_act_logits = hoi_act_repr @ hoi_act_emb.t()

        hoi_repr = self.app_to_repr_mlps['vp'](torch.cat([hoi_subj_appearance, hoi_obj_appearance, spatial_info], dim=1))
        hoi_repr = torch.nn.functional.normalize(hoi_repr)
        hoi_emb = self.wemb_to_repr_mlps['vp'](self.visual_phrases_embs)
        hoi_logits = hoi_repr @ hoi_emb.t()

        return hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits
