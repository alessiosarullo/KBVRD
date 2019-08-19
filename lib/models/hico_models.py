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
from lib.models.misc import bce_loss, interactions_to_actions, interactions_to_objects, interactions_to_mat


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
        if return_repr:
            return obj_repr, act_repr, hoi_repr
        return obj_logits, act_logits, hoi_logits


class HicoExtZSGCMultiModel(AbstractModel):
    @classmethod
    def get_cline_name(cls):
        return 'hicoall'

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
            seen_obj_inds = pickle.load(open(cfg.active_classes_file, 'rb'))[Splits.TRAIN.value]['obj']
            unseen_obj_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_object_classes)) - set(seen_obj_inds.tolist())))
            self.seen_obj_inds = nn.Parameter(torch.tensor(seen_obj_inds), requires_grad=False)
            self.unseen_obj_inds = nn.Parameter(torch.tensor(unseen_obj_inds), requires_grad=False)

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

        self.multi_model = HicoMultiModel(dataset, **kwargs)
        self.predictor_dim = self.multi_model.repr_dim

        if cfg.gc:
            gcemb_dim = 1024

            if cfg.puregc:
                gc_dims = (gcemb_dim, self.predictor_dim)
            else:
                latent_dim = 200
                input_dim = self.predictor_dim
                self.obj_to_predictor = nn.Sequential(nn.Linear(latent_dim, 600),
                                                      nn.ReLU(inplace=True),
                                                      nn.Dropout(p=cfg.dropout),
                                                      nn.Linear(600, 800),
                                                      nn.ReLU(inplace=True),
                                                      nn.Dropout(p=cfg.dropout),
                                                      nn.Linear(800, input_dim),
                                                      )
                self.act_to_predictor = nn.Sequential(nn.Linear(latent_dim, 600),
                                                      nn.ReLU(inplace=True),
                                                      nn.Dropout(p=cfg.dropout),
                                                      nn.Linear(600, 800),
                                                      nn.ReLU(inplace=True),
                                                      nn.Dropout(p=cfg.dropout),
                                                      nn.Linear(800, input_dim),
                                                      )
                self.hoi_to_predictor = nn.Sequential(nn.Linear(latent_dim, 600),
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

    def get_soft_labels(self, labels):
        assert cfg.zso or cfg.zsa or cfg.zsh
        batch_size = labels.shape[0]
        labels = labels.clamp(min=0)

        inter_mat = interactions_to_mat(labels, hico=self.dataset.full_dataset)  # N x I -> N x O x P
        ext_inter_mat = inter_mat

        obj_labels = interactions_to_objects(labels, hico=self.dataset.full_dataset)
        if cfg.zso:
            obj_emb_sim = self.obj_emb_sim.clamp(min=0)
            similar_obj_per_act = torch.bmm(inter_mat.transpose(1, 2), obj_emb_sim.unsqueeze(dim=0).expand(batch_size, -1, -1)).transpose(2, 1)
            similar_obj_per_act = similar_obj_per_act / inter_mat.sum(dim=1, keepdim=True).clamp(min=1)
            similar_obj_per_act = similar_obj_per_act * self.obj_act_feasibility.unsqueeze(dim=0).expand(batch_size, -1, -1)
            ext_inter_mat += similar_obj_per_act

            if cfg.sloo:
                similar_obj = (obj_labels @ obj_emb_sim).clamp(max=1)
                ext_inter_mat = torch.max(ext_inter_mat, similar_obj.unsqueeze(dim=2)) * \
                                self.obj_act_feasibility.unsqueeze(dim=0).expand(batch_size, -1, -1)

            obj_labels[:, self.unseen_obj_inds] = similar_obj_per_act.max(dim=2)[0][:, self.unseen_obj_inds]
        obj_labels = obj_labels.detach()

        act_labels = interactions_to_actions(labels, hico=self.dataset.full_dataset)
        if cfg.zsa:
            act_emb_sims = self.act_emb_sim.clamp(min=0)
            similar_acts_per_obj = torch.bmm(ext_inter_mat, act_emb_sims.unsqueeze(dim=0).expand(batch_size, -1, -1))
            similar_acts_per_obj = similar_acts_per_obj / ext_inter_mat.sum(dim=2, keepdim=True).clamp(min=1)
            feasible_similar_acts_per_obj = similar_acts_per_obj * self.obj_act_feasibility.unsqueeze(dim=0).expand(batch_size, -1, -1)
            act_labels[:, self.unseen_act_inds] = feasible_similar_acts_per_obj.max(dim=1)[0][:, self.unseen_act_inds]
        act_labels = act_labels.detach()

        hoi_labels = labels
        if cfg.zsh:
            obj_emb_sim = self.obj_emb_sim.clamp(min=0)
            similar_obj_per_act = torch.bmm(ext_inter_mat.transpose(1, 2), obj_emb_sim.unsqueeze(dim=0).expand(batch_size, -1, -1)).transpose(2, 1)
            similar_obj_per_act = similar_obj_per_act / ext_inter_mat.sum(dim=1, keepdim=True).clamp(min=1)

            act_emb_sims = self.act_emb_sim.clamp(min=0)
            similar_acts_per_obj = torch.bmm(ext_inter_mat, act_emb_sims.unsqueeze(dim=0).expand(batch_size, -1, -1))
            similar_acts_per_obj = similar_acts_per_obj / ext_inter_mat.sum(dim=2, keepdim=True).clamp(min=1)

            similar_hois = (similar_obj_per_act + similar_acts_per_obj) * self.obj_act_feasibility.unsqueeze(dim=0).expand(batch_size, -1, -1)

            interactions = self.dataset.full_dataset.interactions  # FIXME should do based on graph instead of oracle
            hoi_labels[:, self.unseen_hoi_inds] = similar_hois[:, interactions[:, 1], interactions[:, 0]][:, self.unseen_hoi_inds]
        hoi_labels = hoi_labels.detach()

        return obj_labels, act_labels, hoi_labels

    def get_reg_loss(self, predictors, adj_mat):
        predictors_norm = F.normalize(predictors, dim=1)
        predictors_sim = predictors_norm @ predictors_norm.t()
        arange = torch.arange(predictors_sim.shape[0])

        # Done with argmin/argmax because using min/max directly resulted in NaNs.
        neigh_mask = torch.full_like(predictors_sim, np.inf)
        neigh_mask[adj_mat] = 1
        argmin_neigh_sim = (predictors_sim * neigh_mask.detach()).argmin(dim=1)
        min_neigh_sim = predictors_sim[arange, argmin_neigh_sim]

        non_neigh_mask = torch.full_like(predictors_sim, -np.inf)
        non_neigh_mask[~adj_mat] = 1
        argmax_non_neigh_sim = (predictors_sim * non_neigh_mask.detach()).argmax(dim=1)
        max_non_neigh_sim = predictors_sim[arange, argmax_non_neigh_sim]

        # Exclude null interaction
        min_neigh_sim = min_neigh_sim[1:]
        max_non_neigh_sim = max_non_neigh_sim[1:]

        assert not torch.isinf(min_neigh_sim).any() and not torch.isinf(max_non_neigh_sim).any()
        assert not torch.isnan(min_neigh_sim).any() and not torch.isnan(max_non_neigh_sim).any()

        reg_loss_mat = F.relu(cfg.greg_margin - min_neigh_sim + max_non_neigh_sim)
        reg_loss = reg_loss_mat.mean()
        return reg_loss

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):

            feats, labels = x
            obj_logits, act_logits, hoi_logits, obj_labels, act_labels, hoi_labels, reg_loss = self._forward(feats, labels)

            if not inference:
                obj_labels.clamp_(min=0), act_labels.clamp_(min=0), hoi_labels.clamp_(min=0)
                if self.zs_enabled:
                    losses = {'obj_loss': cfg.olc * bce_loss(obj_logits[:, self.seen_obj_inds], obj_labels[:, self.seen_obj_inds]),
                              'act_loss': cfg.alc * bce_loss(act_logits[:, self.seen_act_inds], act_labels[:, self.seen_act_inds]),
                              'hoi_loss': cfg.hlc * bce_loss(hoi_logits[:, self.seen_hoi_inds], hoi_labels[:, self.seen_hoi_inds]),
                              }
                    if cfg.softl > 0:
                        if cfg.zso:
                            losses['obj_loss_unseen'] = cfg.olc * cfg.softl * bce_loss(obj_logits[:, self.unseen_obj_inds],
                                                                                       obj_labels[:, self.unseen_obj_inds])
                        if cfg.zsa:
                            losses['act_loss_unseen'] = cfg.alc * cfg.softl * bce_loss(act_logits[:, self.unseen_act_inds],
                                                                                       act_labels[:, self.unseen_act_inds])
                        if cfg.zsh:
                            losses['hoi_loss_unseen'] = cfg.hlc * cfg.softl * bce_loss(hoi_logits[:, self.unseen_hoi_inds],
                                                                                       hoi_labels[:, self.unseen_hoi_inds])
                else:
                    losses = {'obj_loss': cfg.olc * bce_loss(obj_logits, obj_labels),
                              'act_loss': cfg.alc * bce_loss(act_logits, act_labels),
                              'hoi_loss': cfg.hlc * bce_loss(hoi_logits, hoi_labels),
                              }
                if reg_loss is not None:
                    losses['reg_loss'] = reg_loss
                return losses
            else:
                prediction = Prediction()
                interactions = self.dataset.full_dataset.interactions
                obj_scores = torch.sigmoid(obj_logits).cpu().numpy() ** cfg.osc
                act_scores = torch.sigmoid(act_logits).cpu().numpy() ** cfg.asc
                hoi_scores = torch.sigmoid(hoi_logits).cpu().numpy() ** cfg.hsc
                prediction.hoi_scores = obj_scores[:, interactions[:, 1]] * act_scores[:, interactions[:, 0]] * hoi_scores
                return prediction

    def _forward(self, feats, labels):
        if labels is not None:
            hoi_labels = labels.clamp(min=0)
            obj_labels = interactions_to_objects(hoi_labels, self.dataset.full_dataset).detach()
            act_labels = interactions_to_actions(hoi_labels, self.dataset.full_dataset).detach()
            if cfg.softl > 0:
                obj_labels, act_labels, hoi_labels = self.get_soft_labels(labels)
        else:
            obj_labels = act_labels = hoi_labels = None

        reg_loss = None
        if cfg.gc:
            obj_repr, act_repr, hoi_repr = self.multi_model._forward(feats, labels, return_repr=True)
            if cfg.hoigcn:
                hoi_class_embs, act_class_embs, obj_class_embs = self.gcn()
            else:
                hico = self.dataset.full_dataset
                obj_class_embs, act_class_embs = self.gcn()  # P x E
                hoi_class_embs = F.normalize(obj_class_embs[hico.interactions[:, 1]] + act_class_embs[hico.interactions[:, 0]], dim=1)

            if not cfg.puregc:
                obj_predictors = self.obj_to_predictor(obj_class_embs)  # P x D
                act_predictors = self.act_to_predictor(act_class_embs)  # P x D
                hoi_predictors = self.hoi_to_predictor(hoi_class_embs)  # P x D
            else:
                obj_predictors = obj_class_embs
                act_predictors = act_class_embs
                hoi_predictors = hoi_class_embs
            obj_logits = obj_repr @ obj_predictors.t()
            act_logits = act_repr @ act_predictors.t()
            hoi_logits = hoi_repr @ hoi_predictors.t()

            if cfg.greg > 0:
                reg_loss = cfg.greg * self.get_reg_loss(hoi_predictors, self.inter_adj)
        else:
            # obj_logits, act_logits, hoi_logits = self.multi_model._forward(feats, labels, return_repr=False)
            obj_repr, act_repr, hoi_repr = self.multi_model._forward(feats, labels, return_repr=True)
            obj_predictor = self.multi_model.obj_output_mlp.weight
            act_predictor = self.multi_model.act_output_mlp.weight
            hoi_predictor = self.multi_model.hoi_output_mlp.weight
            obj_logits = obj_repr @ obj_predictor.t()
            act_logits = act_repr @ act_predictor.t()
            hoi_logits = hoi_repr @ hoi_predictor.t()
        return obj_logits, act_logits, hoi_logits, obj_labels, act_labels, hoi_labels, reg_loss


class KatoModel(AbstractModel):
    @classmethod
    def get_cline_name(cls):
        return 'kato'

    def __init__(self, dataset: HicoSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        super().__init__(dataset, **kwargs)
        assert cfg.hico
        self.dataset = dataset

        assert cfg.seenf >= 0  # ZS enabled
        seen_pred_inds = pickle.load(open(cfg.active_classes_file, 'rb'))[Splits.TRAIN.value]['pred']
        seen_op_mat = dataset.full_dataset.op_pair_to_interaction[:, seen_pred_inds]
        seen_hoi_inds = np.sort(seen_op_mat[seen_op_mat >= 0])
        unseen_hoi_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_interactions)) - set(seen_hoi_inds.tolist())))
        self.seen_hoi_inds = nn.Parameter(torch.tensor(seen_hoi_inds), requires_grad=False)
        self.unseen_hoi_inds = nn.Parameter(torch.tensor(unseen_hoi_inds), requires_grad=False)

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

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):

            feats, labels = x
            logits, labels = self._forward(feats, labels)

            if not inference:
                labels.clamp_(min=0)
                losses = {'hoi_loss': bce_loss(logits[:, self.seen_hoi_inds], labels[:, self.seen_hoi_inds])}
                return losses
            else:
                prediction = Prediction()
                prediction.hoi_scores = torch.sigmoid(logits).cpu().numpy()
                return prediction

    def _forward(self, feats, labels):
        hoi_repr = self.img_repr_mlp(feats)
        z_a, z_v, z_n = self.gcn_branch()
        hoi_logits = self.score_mlp(torch.cat([hoi_repr.unsqueeze(dim=1).expand(-1, z_a.shape[0], -1),
                                               z_a.unsqueeze(dim=0).expand(hoi_repr.shape[0], -1, -1)],
                                              dim=2))
        assert hoi_logits.shape[2] == 1
        hoi_logits = hoi_logits.squeeze(dim=2)
        return hoi_logits, labels
