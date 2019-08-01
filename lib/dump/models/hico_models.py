import pickle
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from lib.dataset.hico.hico_split import HicoSplit
from lib.dataset.utils import Splits
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.abstract_model import AbstractModel
from lib.models.branches import get_noun_verb_adj_mat, CheatGCNBranch, CheatHoiGCNBranch, KatoGCNBranch
from lib.models.containers import Prediction
from lib.models.misc import bce_loss, LIS, interactions_to_actions, interactions_to_objects, interactions_to_mat


class HicoBaseModel(AbstractModel):
    @classmethod
    def get_cline_name(cls):
        return 'hicobase'

    def __init__(self, dataset: HicoSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        assert cfg.hico
        self.dataset = dataset

        vis_feat_dim = self.dataset.precomputed_visual_feat_dim
        hidden_dim = 1024
        if cfg.small:
            self.repr_dim = 512
        else:
            self.repr_dim = 1024

        self.hoi_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.dropout),
                                            nn.Linear(hidden_dim, self.repr_dim),
                                            ])
        nn.init.xavier_normal_(self.hoi_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.hoi_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))

        self.output_mlp = nn.Linear(self.repr_dim, dataset.full_dataset.num_interactions, bias=False)
        torch.nn.init.xavier_normal_(self.output_mlp.weight, gain=1.0)

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):

            if cfg.mlneg:
                feats, labels, label_mask = x
            else:
                feats, labels = x
                label_mask = None
            output = self._forward(feats, labels)

            if not inference:
                zero_labels = (labels == 0)
                labels.clamp_(min=0)
                loss_mat = bce_loss(output, labels, reduce=False)
                if cfg.hico_lhard:
                    loss_mat[zero_labels] = 0
                # if cfg.iso_null:
                #     null_interactions = np.flatnonzero(self.dataset.full_dataset.interactions[:, 0] == 0)
                #     loss_mat[:, null_interactions] = 0
                if cfg.mlneg:
                    losses = {'hoi_loss': loss_mat[label_mask].sum() / loss_mat.shape[0]}
                else:
                    losses = {'hoi_loss': loss_mat.sum(dim=1).mean()}
                return losses
            else:
                prediction = Prediction()
                prediction.hoi_scores = torch.sigmoid(output).cpu().numpy()
                return prediction

    def _forward(self, feats, labels, return_repr=False):
        hoi_repr = self.hoi_repr_mlp(feats)
        if return_repr:
            return hoi_repr
        else:
            output_logits = self.output_mlp(hoi_repr)
            return output_logits


class HicoMultiModel(AbstractModel):
    @classmethod
    def get_cline_name(cls):
        return 'hicomulti'

    def __init__(self, dataset: HicoSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        assert cfg.hico
        self.dataset = dataset

        vis_feat_dim = self.dataset.precomputed_visual_feat_dim
        hidden_dim = 1024
        self.repr_dim = 1024

        self.obj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.dropout),
                                            nn.Linear(hidden_dim, self.repr_dim),
                                            ])
        nn.init.xavier_normal_(self.obj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.obj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))
        self.obj_output_mlp = nn.Linear(self.repr_dim, dataset.full_dataset.num_object_classes, bias=False)
        torch.nn.init.xavier_normal_(self.obj_output_mlp.weight, gain=1.0)

        self.act_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.dropout),
                                            nn.Linear(hidden_dim, self.repr_dim),
                                            ])
        nn.init.xavier_normal_(self.act_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.act_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))
        self.act_output_mlp = nn.Linear(self.repr_dim, dataset.full_dataset.num_actions, bias=False)
        torch.nn.init.xavier_normal_(self.act_output_mlp.weight, gain=1.0)

        self.hoi_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.dropout),
                                            nn.Linear(hidden_dim, self.repr_dim),
                                            ])
        nn.init.xavier_normal_(self.hoi_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.hoi_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))
        self.hoi_output_mlp = nn.Linear(self.repr_dim, dataset.full_dataset.num_interactions, bias=False)
        torch.nn.init.xavier_normal_(self.hoi_output_mlp.weight, gain=1.0)

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):

            feats, labels = x
            obj_logits, act_logits, hoi_logits = self._forward(feats, labels)

            if not inference:
                hoi_labels = labels.clamp(min=0)
                obj_labels = interactions_to_objects(hoi_labels, self.dataset.full_dataset).detach()
                act_labels = interactions_to_actions(hoi_labels, self.dataset.full_dataset).detach()
                losses = {'obj_loss': bce_loss(obj_logits, obj_labels),
                          'act_loss': bce_loss(act_logits, act_labels),
                          'hoi_loss': bce_loss(hoi_logits, hoi_labels)
                          }
                return losses
            else:
                prediction = Prediction()
                interactions = self.dataset.full_dataset.interactions
                obj_scores = torch.sigmoid(obj_logits).cpu().numpy()
                act_scores = torch.sigmoid(act_logits).cpu().numpy()
                hoi_scores = torch.sigmoid(hoi_logits).cpu().numpy()
                prediction.hoi_scores = obj_scores[:, interactions[:, 1]] * act_scores[:, interactions[:, 0]] * hoi_scores
                return prediction

    def _forward(self, feats, labels, return_repr=False):
        obj_repr = self.obj_repr_mlp(feats)
        obj_logits = self.obj_output_mlp(obj_repr)

        act_repr = self.act_repr_mlp(feats)
        act_logits = self.act_output_mlp(act_repr)

        hoi_repr = self.hoi_repr_mlp(feats)
        hoi_logits = self.hoi_output_mlp(hoi_repr)
        return obj_logits, act_logits, hoi_logits


class HicoExtKnowledgeGenericModel(AbstractModel):
    def __init__(self, dataset: HicoSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        assert cfg.hico
        self.dataset = dataset

        self.nv_adj = get_noun_verb_adj_mat(dataset=dataset, iso_null=True)

        word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        obj_wembs = word_embs.get_embeddings(dataset.full_dataset.objects, retry='avg')
        act_wembs = word_embs.get_embeddings(dataset.full_dataset.predicates, retry='avg')
        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_wembs), requires_grad=False)
        self.obj_emb_sim = nn.Parameter(self.obj_word_embs @ self.obj_word_embs.t(), requires_grad=False)
        self.act_word_embs = nn.Parameter(torch.from_numpy(act_wembs), requires_grad=False)
        self.act_emb_sim = nn.Parameter(self.act_word_embs @ self.act_word_embs.t(), requires_grad=False)
        self.act_obj_emb_sim = nn.Parameter(self.act_word_embs @ self.obj_word_embs.t(), requires_grad=False)

        self.zs_enabled = (cfg.seenf >= 0)
        self.load_backbone = len(cfg.hoi_backbone) > 0
        if self.zs_enabled:
            print('Zero-shot enabled.')
            seen_pred_inds = pickle.load(open(cfg.active_classes_file, 'rb'))[Splits.TRAIN.value]['pred']

            seen_op_mat = dataset.full_dataset.op_pair_to_interaction[:, seen_pred_inds]
            seen_hoi_inds = np.sort(seen_op_mat[seen_op_mat >= 0])
            unseen_hoi_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_interactions)) - set(seen_hoi_inds.tolist())))
            self.seen_hoi_inds = nn.Parameter(torch.tensor(seen_hoi_inds), requires_grad=False)
            self.unseen_hoi_inds = nn.Parameter(torch.tensor(unseen_hoi_inds), requires_grad=False)

            if self.load_backbone:
                raise NotImplementedError('Not currently supported.')

            if cfg.softl:
                self.obj_act_feasibility = nn.Parameter(self.nv_adj, requires_grad=False)

    def get_soft_labels(self, labels):
        batch_size = labels.shape[0]

        inter_mat = interactions_to_mat(labels.clamp(min=0), hico=self.dataset.full_dataset)  # N x I -> N x O x P

        similar_objs = None
        if cfg.hico_zso1:
            objects = interactions_to_objects(labels, hico=self.dataset.full_dataset)
            similar_objs = objects @ self.obj_emb_sim

        if cfg.hico_zso2:
            actions = interactions_to_actions(labels, hico=self.dataset.full_dataset)
            similar_objs_by_act = actions @ self.act_obj_emb_sim.clamp(min=0) / actions.sum(dim=1, keepdim=True).clamp(min=1)
            if cfg.hico_zso1:
                similar_objs = (similar_objs + similar_objs_by_act) / 2
            else:
                similar_objs = similar_objs_by_act

        if similar_objs is not None:
            similar_objs = similar_objs.unsqueeze(dim=2).expand(-1, -1, self.obj_act_feasibility.shape[1])
            possible_interactions_by_obj = similar_objs * self.obj_act_feasibility.unsqueeze(dim=0).expand_as(similar_objs)
            inter_mat = inter_mat + (1 - inter_mat) * possible_interactions_by_obj

        similar_acts_per_obj = torch.bmm(inter_mat, self.act_emb_sim.unsqueeze(dim=0).clamp(min=0).expand(batch_size, -1, -1))
        if cfg.slnoavg:
            similar_acts_per_obj = similar_acts_per_obj.clamp(max=1)
        else:
            similar_acts_per_obj = similar_acts_per_obj / inter_mat.sum(dim=2, keepdim=True).clamp(min=1)

        if cfg.lis:
            if cfg.hardlis:
                w, k = 30, 18
            else:
                w, k = 18, 7
            similar_acts_per_obj = LIS(similar_acts_per_obj, w=w, k=k)

        interactions = self.dataset.full_dataset.interactions
        unseen_labels = similar_acts_per_obj[:, interactions[:, 1], interactions[:, 0]][:, self.unseen_hoi_inds]
        return unseen_labels.detach()

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):

            feats, labels = x
            logits, labels, reg_loss = self._forward(feats, labels)

            if not inference:
                labels.clamp_(min=0)
                losses = {'hoi_loss': bce_loss(logits[:, self.seen_hoi_inds], labels[:, self.seen_hoi_inds])}
                if cfg.softl > 0:
                    # if cfg.nullzs:
                    #     unseen_action_labels *= (1 - action_labels[:, :1])  # cannot be anything else if it is a positive (i.e., from GT) null
                    losses['hoi_loss_unseen'] = cfg.softl * bce_loss(logits[:, self.unseen_hoi_inds], labels[:, self.unseen_hoi_inds])
                if reg_loss is not None:
                    losses['reg_loss'] = reg_loss
                return losses
            else:
                prediction = Prediction()
                prediction.hoi_scores = torch.sigmoid(logits).cpu().numpy()
                return prediction


class HicoExtKnowledgeMultiGenericModel(AbstractModel):
    def __init__(self, dataset: HicoSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        assert cfg.hico
        self.dataset = dataset

        self.nv_adj = get_noun_verb_adj_mat(dataset=dataset, iso_null=True)

        word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        obj_wembs = word_embs.get_embeddings(dataset.full_dataset.objects, retry='avg')
        act_wembs = word_embs.get_embeddings(dataset.full_dataset.predicates, retry='avg')
        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_wembs), requires_grad=False)
        self.obj_emb_sim = nn.Parameter(self.obj_word_embs @ self.obj_word_embs.t(), requires_grad=False)
        self.act_word_embs = nn.Parameter(torch.from_numpy(act_wembs), requires_grad=False)
        self.act_emb_sim = nn.Parameter(self.act_word_embs @ self.act_word_embs.t(), requires_grad=False)
        self.act_obj_emb_sim = nn.Parameter(self.act_word_embs @ self.obj_word_embs.t(), requires_grad=False)

        self.zs_enabled = (cfg.seenf >= 0)
        self.load_backbone = len(cfg.hoi_backbone) > 0
        if self.zs_enabled:
            print('Zero-shot enabled.')
            seen_act_inds = pickle.load(open(cfg.active_classes_file, 'rb'))[Splits.TRAIN.value]['pred']
            unseen_act_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_actions)) - set(seen_act_inds.tolist())))
            self.seen_act_inds = nn.Parameter(torch.tensor(seen_act_inds), requires_grad=False)
            self.unseen_act_inds = nn.Parameter(torch.tensor(unseen_act_inds), requires_grad=False)

            seen_op_mat = dataset.full_dataset.op_pair_to_interaction[:, seen_act_inds]
            seen_hoi_inds = np.sort(seen_op_mat[seen_op_mat >= 0])
            unseen_hoi_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_interactions)) - set(seen_hoi_inds.tolist())))
            self.seen_hoi_inds = nn.Parameter(torch.tensor(seen_hoi_inds), requires_grad=False)
            self.unseen_hoi_inds = nn.Parameter(torch.tensor(unseen_hoi_inds), requires_grad=False)

            if self.load_backbone:
                raise NotImplementedError('Not currently supported.')

            if cfg.softl:
                self.obj_act_feasibility = nn.Parameter(self.nv_adj, requires_grad=False)

    def get_soft_labels(self, labels):
        batch_size = labels.shape[0]
        labels = labels.clamp(min=0)

        inter_mat = interactions_to_mat(labels, hico=self.dataset.full_dataset)  # N x I -> N x O x P

        act_emb_sims = self.act_emb_sim.clamp(min=0)
        if cfg.lis:
            act_emb_sims = LIS(act_emb_sims, w=18, k=7)

        similar_acts_per_obj = torch.bmm(inter_mat, act_emb_sims.unsqueeze(dim=0).expand(batch_size, -1, -1))
        similar_acts_per_obj = similar_acts_per_obj / inter_mat.sum(dim=2, keepdim=True).clamp(min=1)

        if cfg.hico_zsa:
            similar_acts_per_obj = similar_acts_per_obj * self.obj_act_feasibility.unsqueeze(dim=0).expand(batch_size, -1, -1)
            unseen_labels = similar_acts_per_obj.max(dim=1)[0][:, self.unseen_act_inds]
        else:
            interactions = self.dataset.full_dataset.interactions
            unseen_labels = similar_acts_per_obj[:, interactions[:, 1], interactions[:, 0]][:, self.unseen_hoi_inds]
        return unseen_labels.detach()

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):

            feats, labels = x
            obj_logits, act_logits, hoi_logits, obj_labels, act_labels, hoi_labels, reg_loss = self._forward(feats, labels)

            if not inference:
                obj_labels.clamp_(min=0), act_labels.clamp_(min=0), hoi_labels.clamp_(min=0)
                if cfg.softl > 0:
                    if cfg.hico_zsa:
                        losses = {'act_loss': bce_loss(act_logits[:, self.seen_act_inds], act_labels[:, self.seen_act_inds]),
                                  'act_loss_unseen': cfg.softl * bce_loss(act_logits[:, self.unseen_act_inds], act_labels[:, self.unseen_act_inds]),
                                  'hoi_loss': bce_loss(hoi_logits, hoi_labels),
                                  }
                    else:
                        losses = {'act_loss': bce_loss(act_logits, act_labels),
                                  'hoi_loss': bce_loss(hoi_logits[:, self.seen_hoi_inds], hoi_labels[:, self.seen_hoi_inds]),
                                  'hoi_loss_unseen': cfg.softl * bce_loss(hoi_logits[:, self.unseen_hoi_inds], hoi_labels[:, self.unseen_hoi_inds])
                                  }
                losses['obj_loss'] = bce_loss(obj_logits, obj_labels)
                if reg_loss is not None:
                    losses['reg_loss'] = reg_loss
                return losses
            else:
                prediction = Prediction()
                interactions = self.dataset.full_dataset.interactions
                obj_scores = torch.sigmoid(obj_logits).cpu().numpy()
                act_scores = torch.sigmoid(act_logits).cpu().numpy()
                hoi_scores = torch.sigmoid(hoi_logits).cpu().numpy()
                prediction.hoi_scores = obj_scores[:, interactions[:, 1]] * act_scores[:, interactions[:, 0]] * hoi_scores
                return prediction


class HicoZSMultiModel(HicoExtKnowledgeMultiGenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'hicozsm'

    def __init__(self, dataset: HicoSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.base_model = HicoMultiModel(dataset, **kwargs)
        assert self.zs_enabled and cfg.softl > 0

    def _forward(self, feats, labels):
        obj_logits, act_logits, hoi_logits = self.base_model._forward(feats, labels)
        if labels is not None:
            hoi_labels = labels.clamp(min=0)
            obj_labels = interactions_to_objects(hoi_labels, self.dataset.full_dataset).detach()
            act_labels = interactions_to_actions(hoi_labels, self.dataset.full_dataset).detach()
            if cfg.hico_zsa:
                act_labels[:, self.unseen_act_inds] = self.get_soft_labels(labels)
            else:
                hoi_labels[:, self.unseen_hoi_inds] = self.get_soft_labels(labels)
        else:
            obj_labels = act_labels = hoi_labels = None
        reg_loss = None
        return obj_logits, act_logits, hoi_logits, obj_labels, act_labels, hoi_labels, reg_loss


class HicoZSBaseModel(HicoExtKnowledgeGenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'hicozsb'

    def __init__(self, dataset: HicoSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.base_model = HicoBaseModel(dataset, **kwargs)
        assert self.zs_enabled and cfg.softl > 0

    def _forward(self, feats, labels):
        logits = self.base_model._forward(feats, labels)
        if labels is not None:
            labels[:, self.unseen_hoi_inds] = self.get_soft_labels(labels)
        reg_loss = None
        return logits, labels, reg_loss


class HicoZSGCModel(HicoExtKnowledgeGenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'hicozsgc'

    def __init__(self, dataset: HicoSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.base_model = HicoBaseModel(dataset, **kwargs)
        self.predictor_dim = self.base_model.repr_dim

        gcemb_dim = 1024
        if cfg.puregc:
            gc_dims = (gcemb_dim, self.predictor_dim)
        else:
            latent_dim = 200
            input_dim = self.predictor_dim
            if cfg.small:
                self.emb_to_predictor = nn.Sequential(nn.Linear(latent_dim, 300),
                                                      nn.ReLU(inplace=True),
                                                      nn.Dropout(p=cfg.dropout),
                                                      nn.Linear(300, 400),
                                                      nn.ReLU(inplace=True),
                                                      nn.Dropout(p=cfg.dropout),
                                                      nn.Linear(400, input_dim),
                                                      )
            else:
                self.emb_to_predictor = nn.Sequential(nn.Linear(latent_dim, 600),
                                                      nn.ReLU(inplace=True),
                                                      nn.Dropout(p=cfg.dropout),
                                                      nn.Linear(600, 800),
                                                      nn.ReLU(inplace=True),
                                                      nn.Dropout(p=cfg.dropout),
                                                      nn.Linear(800, input_dim),
                                                      )
            gc_dims = (gcemb_dim // 2, latent_dim)

        if cfg.hoigcn:
            self.gcn = CheatHoiGCNBranch(dataset, input_repr_dim=gcemb_dim, gc_dims=gc_dims)
        else:
            self.gcn = CheatGCNBranch(dataset, input_repr_dim=gcemb_dim, gc_dims=gc_dims)

        if cfg.greg > 0:
            inter_nv_adj = np.zeros((self.dataset.full_dataset.num_interactions,
                                     self.dataset.full_dataset.num_object_classes + self.dataset.full_dataset.num_actions))
            interactions = self.dataset.full_dataset.interactions
            inter_nv_adj[np.arange(interactions.shape[0]), interactions[:, 0]] = 1
            inter_nv_adj[np.arange(interactions.shape[0]), interactions[:, 1]] = 1
            self.inter_adj = nn.Parameter(torch.from_numpy(inter_nv_adj @ inter_nv_adj.T).clamp(max=1).byte(), requires_grad=False)
            assert (self.inter_adj.diag()[1:] == 1).all()

    def _forward(self, feats, labels):
        vrepr = self.base_model._forward(feats, labels, return_repr=True)

        hico = self.dataset.full_dataset
        if cfg.hoigcn:
            hoi_class_embs, _, _ = self.gcn()
        else:
            obj_class_embs, act_class_embs = self.gcn()  # P x E
            hoi_class_embs = F.normalize(obj_class_embs[hico.interactions[:, 1]] + act_class_embs[hico.interactions[:, 0]], dim=1)

        if not cfg.puregc:
            hoi_predictors = self.emb_to_predictor(hoi_class_embs)  # P x D
        else:
            hoi_predictors = hoi_class_embs
        logits = vrepr @ hoi_predictors.t()

        if labels is not None and self.zs_enabled:
            if cfg.softl > 0:
                labels[:, self.unseen_hoi_inds] = self.get_soft_labels(labels)

        reg_loss = None
        if cfg.greg > 0:
            hoi_predictors_norm = F.normalize(hoi_predictors, dim=1)
            hoi_predictors_sim = hoi_predictors_norm @ hoi_predictors_norm.t()
            arange = torch.arange(hoi_predictors_sim.shape[0])

            # Done with argmin/argmax because using min/max directly resulted in NaNs.
            neigh_mask = torch.full_like(hoi_predictors_sim, np.inf)
            neigh_mask[self.inter_adj] = 1
            argmin_neigh_sim = (hoi_predictors_sim * neigh_mask.detach()).argmin(dim=1)
            min_neigh_sim = hoi_predictors_sim[arange, argmin_neigh_sim]

            non_neigh_mask = torch.full_like(hoi_predictors_sim, -np.inf)
            non_neigh_mask[~self.inter_adj] = 1
            argmax_non_neigh_sim = (hoi_predictors_sim * non_neigh_mask.detach()).argmax(dim=1)
            max_non_neigh_sim = hoi_predictors_sim[arange, argmax_non_neigh_sim]

            # Exclude null interaction
            min_neigh_sim = min_neigh_sim[1:]
            max_non_neigh_sim = max_non_neigh_sim[1:]

            assert not torch.isinf(min_neigh_sim).any() and not torch.isinf(max_non_neigh_sim).any()
            assert not torch.isnan(min_neigh_sim).any() and not torch.isnan(max_non_neigh_sim).any()

            reg_loss = F.relu(cfg.greg_margin - min_neigh_sim + max_non_neigh_sim)
            reg_loss = cfg.greg * reg_loss.mean()

        return logits, labels, reg_loss


class KatoModel(HicoExtKnowledgeGenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'kato'

    def __init__(self, dataset: HicoSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.hoi_repr_dim = 512
        vis_feat_dim = self.dataset.precomputed_visual_feat_dim

        self.img_repr_mlp = nn.Linear(vis_feat_dim, self.hoi_repr_dim)

        gc_dims = (512, 200)
        self.gcn_branch = KatoGCNBranch(dataset, gc_dims=(512, 200))
        self.score_mlp = nn.Sequential(nn.Linear(gc_dims[-1] + self.hoi_repr_dim, 512),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(512, 200),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(p=0.5),
                                       nn.Linear(200, 1)
                                       )

    def _forward(self, feats, labels):
        hoi_repr = self.img_repr_mlp(feats)
        z_a, z_v, z_n = self.gcn_branch()
        hoi_logits = self.score_mlp(torch.cat([hoi_repr.unsqueeze(dim=1).expand(-1, z_a.shape[0], -1),
                                               z_a.unsqueeze(dim=0).expand(hoi_repr.shape[0], -1, -1)],
                                              dim=2))
        assert hoi_logits.shape[2] == 1
        hoi_logits = hoi_logits.squeeze(dim=2)
        return hoi_logits, labels, None
