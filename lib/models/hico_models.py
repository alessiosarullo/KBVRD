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
from lib.models.misc import bce_loss, interactions_to_actions, interactions_to_objects, interactions_to_mat, get_hoi_adjacency_matrix


class HicoExtZSGCMultiModel(AbstractModel):
    @classmethod
    def get_cline_name(cls):
        return 'hicoall'

    def __init__(self, dataset: HicoSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        assert cfg.hico
        self.dataset = dataset
        self.repr_dim = 1024
        self.loss_coeffs = {'obj': cfg.olc, 'act': cfg.alc, 'hoi': cfg.hlc}
        self.reg_coeffs = {'obj': cfg.pro, 'act': cfg.pra, 'hoi': cfg.prh}

        # Object-action (or noun-verb) adjacency matrix
        self.nv_adj = get_noun_verb_adj_mat(dataset=dataset, isolate_null=True)

        # Word embeddings + similarity matrices
        word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        obj_wembs = word_embs.get_embeddings(dataset.full_dataset.objects, retry='avg')
        act_wembs = word_embs.get_embeddings(dataset.full_dataset.predicates, retry='avg')
        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_wembs), requires_grad=False)
        self.obj_emb_sim = nn.Parameter(self.obj_word_embs @ self.obj_word_embs.t(), requires_grad=False)
        self.act_word_embs = nn.Parameter(torch.from_numpy(act_wembs), requires_grad=False)
        self.act_emb_sim = nn.Parameter(self.act_word_embs @ self.act_word_embs.t(), requires_grad=False)
        self.act_obj_emb_sim = nn.Parameter(self.act_word_embs @ self.obj_word_embs.t(), requires_grad=False)

        # Seen/unseen indices
        self.zs_enabled = (cfg.seenf >= 0)
        self.load_backbone = len(cfg.hoi_backbone) > 0
        if self.zs_enabled:
            print('Zero-shot enabled.')
            self.seen_inds = {}
            self.unseen_inds = {}

            seen_obj_inds = pickle.load(open(cfg.active_classes_file, 'rb'))[Splits.TRAIN.value]['obj']
            unseen_obj_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_object_classes)) - set(seen_obj_inds.tolist())))
            self.seen_inds['obj'] = nn.Parameter(torch.tensor(seen_obj_inds), requires_grad=False)
            self.unseen_inds['obj'] = nn.Parameter(torch.tensor(unseen_obj_inds), requires_grad=False)

            seen_act_inds = pickle.load(open(cfg.active_classes_file, 'rb'))[Splits.TRAIN.value]['pred']
            unseen_act_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_actions)) - set(seen_act_inds.tolist())))
            self.seen_inds['act'] = nn.Parameter(torch.tensor(seen_act_inds), requires_grad=False)
            self.unseen_inds['act'] = nn.Parameter(torch.tensor(unseen_act_inds), requires_grad=False)

            seen_op_mat = dataset.full_dataset.op_pair_to_interaction[:, seen_act_inds]
            seen_hoi_inds = np.sort(seen_op_mat[seen_op_mat >= 0])
            unseen_hoi_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_interactions)) - set(seen_hoi_inds.tolist())))
            self.seen_inds['hoi'] = nn.Parameter(torch.tensor(seen_hoi_inds), requires_grad=False)
            self.unseen_inds['hoi'] = nn.Parameter(torch.tensor(unseen_hoi_inds), requires_grad=False)

            if self.load_backbone:
                raise NotImplementedError('Not currently supported.')

            if cfg.softl > 0:
                self.softl_enabled = {'obj': cfg.zso, 'act': cfg.zsa, 'hoi': cfg.zsh}
                self.obj_act_feasibility = nn.Parameter(self.nv_adj, requires_grad=False)

        # Base model
        self.repr_mlps = nn.ModuleDict()
        for k in ['obj', 'act', 'hoi']:
            self.repr_mlps[k] = nn.Sequential(*[nn.Linear(self.dataset.precomputed_visual_feat_dim, 1024),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=cfg.dropout),
                                                nn.Linear(1024, self.repr_dim),
                                                ])
            nn.init.xavier_normal_(self.repr_mlps[k][0].weight, gain=torch.nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.repr_mlps[k][3].weight, gain=torch.nn.init.calculate_gain('linear'))

        # Predictors
        if cfg.gc:
            gcemb_dim = 1024
            latent_dim = 200
            self.predictor_mlps = {k: nn.Sequential(nn.Linear(latent_dim, 600),
                                                    nn.ReLU(inplace=True),
                                                    nn.Dropout(p=cfg.dropout),
                                                    nn.Linear(600, 800),
                                                    nn.ReLU(inplace=True),
                                                    nn.Dropout(p=cfg.dropout),
                                                    nn.Linear(800, self.repr_dim),
                                                    ) for k in ['obj', 'act', 'hoi']}
            gc_dims = (gcemb_dim // 2, latent_dim)

            if cfg.hoigcn:
                self.gcn = CheatHoiGCNBranch(dataset, input_repr_dim=gcemb_dim, gc_dims=gc_dims)
            else:
                self.gcn = CheatGCNBranch(dataset, input_repr_dim=gcemb_dim, gc_dims=gc_dims)
        else:
            # Linear predictors (AKA, single FC layer)
            self.output_mlps = nn.ParameterDict()
            for k, d in [('obj', dataset.full_dataset.num_object_classes),
                         ('act', dataset.full_dataset.num_actions),
                         ('hoi', dataset.full_dataset.num_interactions)]:
                self.output_mlps[k] = nn.Parameter(torch.empty(d, self.repr_dim))
                torch.nn.init.xavier_normal_(self.output_mlps[k], gain=1.0)

        # Regularisation
        if self.reg_coeffs['obj'] > 0:
            self.adj_pos['obj'] = nn.Parameter((self.nv_adj @ self.nv_adj.t()).clamp(max=1).byte(), requires_grad=False)
            self.adj_neg['obj'] = nn.Parameter(~self.adj_pos['obj'], requires_grad=False)
        if self.reg_coeffs['act'] > 0:
            self.adj_pos['act'] = nn.Parameter((self.nv_adj.t() @ self.nv_adj).clamp(max=1).byte(), requires_grad=False)
            self.adj_neg['act'] = nn.Parameter(~self.adj_pos['act'], requires_grad=False)
            self.adj_neg['act'][0, :] = 0
            self.adj_neg['act'][:, 0] = 0
        if self.reg_coeffs['hoi'] > 0:
            adj_pos, adj_neg = get_hoi_adjacency_matrix(self.dataset, isolate_null=True)
            self.adj_pos['hoi'] = nn.Parameter(adj_pos, requires_grad=False)
            self.adj_neg['hoi'] = nn.Parameter(adj_neg, requires_grad=False)

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

            obj_labels[:, self.unseen_inds['obj']] = similar_obj_per_act.max(dim=2)[0][:, self.unseen_inds['obj']]
        obj_labels = obj_labels.detach()

        act_labels = interactions_to_actions(labels, hico=self.dataset.full_dataset)
        if cfg.zsa:
            act_emb_sims = self.act_emb_sim.clamp(min=0)
            similar_acts_per_obj = torch.bmm(ext_inter_mat, act_emb_sims.unsqueeze(dim=0).expand(batch_size, -1, -1))
            similar_acts_per_obj = similar_acts_per_obj / ext_inter_mat.sum(dim=2, keepdim=True).clamp(min=1)
            feasible_similar_acts_per_obj = similar_acts_per_obj * self.obj_act_feasibility.unsqueeze(dim=0).expand(batch_size, -1, -1)
            act_labels[:, self.unseen_inds['act']] = feasible_similar_acts_per_obj.max(dim=1)[0][:, self.unseen_inds['act']]
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
            hoi_labels[:, self.unseen_inds['hoi']] = similar_hois[:, interactions[:, 1], interactions[:, 0]][:, self.unseen_inds['hoi']]
        hoi_labels = hoi_labels.detach()

        return obj_labels, act_labels, hoi_labels

    def get_reg_loss(self, predictors, pos_mat, neg_mat):
        predictors_norm = F.normalize(predictors, dim=1)
        predictors_sim = predictors_norm @ predictors_norm.t()
        arange = torch.arange(predictors_sim.shape[0])

        # Done with argmin/argmax because using min/max directly resulted in NaNs.
        neigh_mask = torch.full_like(predictors_sim, np.inf)
        neigh_mask[pos_mat] = 1
        argmin_neigh_sim = (predictors_sim * neigh_mask.detach()).argmin(dim=1)
        min_neigh_sim = predictors_sim[arange, argmin_neigh_sim]

        non_neigh_mask = torch.full_like(predictors_sim, -np.inf)
        non_neigh_mask[neg_mat] = 1
        argmax_non_neigh_sim = (predictors_sim * non_neigh_mask.detach()).argmax(dim=1)
        max_non_neigh_sim = predictors_sim[arange, argmax_non_neigh_sim]

        assert not torch.isinf(min_neigh_sim).any() and not torch.isinf(max_non_neigh_sim).any()
        assert not torch.isnan(min_neigh_sim).any() and not torch.isnan(max_non_neigh_sim).any()

        reg_loss_mat = F.relu(cfg.greg_margin - min_neigh_sim + max_non_neigh_sim)
        reg_loss = reg_loss_mat.mean()
        return reg_loss

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):

            feats, orig_labels = x
            logits, labels, reg_losses = self._forward(feats, orig_labels)

            if not inference:
                labels = {k: v.clamp(min=0) for k, v in labels.items()}
                losses = {}
                for k in ['obj', 'act', 'hoi']:
                    if self.zs_enabled:
                        losses[f'{k}_loss'] = self.loss_coeffs[k] * bce_loss(logits[k][:, self.seen_inds[k]], labels[k][:, self.seen_inds[k]])
                        if cfg.softl > 0 and self.softl_enabled[k]:
                            losses[f'{k}_loss_unseen'] = self.loss_coeffs[k] * cfg.softl * bce_loss(logits[k][:, self.unseen_inds[k]],
                                                                                                    labels[k][:, self.unseen_inds[k]])
                    else:
                        losses[f'{k}_loss'] = self.loss_coeffs[k] * bce_loss(logits[k], labels[k])
                for k, v in reg_losses.items():
                    losses[f'{k}_reg_loss'] = v
                return losses
            else:
                prediction = Prediction()
                interactions = self.dataset.full_dataset.interactions
                obj_scores = torch.sigmoid(logits['obj']).cpu().numpy() ** cfg.osc
                act_scores = torch.sigmoid(logits['act']).cpu().numpy() ** cfg.asc
                hoi_scores = torch.sigmoid(logits['hoi']).cpu().numpy() ** cfg.hsc
                prediction.hoi_scores = obj_scores[:, interactions[:, 1]] * act_scores[:, interactions[:, 0]] * hoi_scores
                return prediction

    def _forward(self, feats, labels):
        # Labels
        if labels is not None:
            hoi_labels = labels.clamp(min=0)
            obj_labels = interactions_to_objects(hoi_labels, self.dataset.full_dataset).detach()
            act_labels = interactions_to_actions(hoi_labels, self.dataset.full_dataset).detach()
            if cfg.softl > 0:
                obj_labels, act_labels, hoi_labels = self.get_soft_labels(labels)
        else:
            obj_labels = act_labels = hoi_labels = None
        labels = {'obj': obj_labels, 'act': act_labels, 'hoi': hoi_labels}

        # Predictors
        if cfg.gc:
            if cfg.hoigcn:
                hoi_class_embs, act_class_embs, obj_class_embs = self.gcn()
            else:
                hico = self.dataset.full_dataset
                obj_class_embs, act_class_embs = self.gcn()  # P x E
                hoi_class_embs = F.normalize(obj_class_embs[hico.interactions[:, 1]] + act_class_embs[hico.interactions[:, 0]], dim=1)
            class_embs = {'obj': obj_class_embs, 'act': act_class_embs, 'hoi': hoi_class_embs}
            predictors = {k: self.predictor_mlps[k](class_embs[k]) for k in ['obj', 'act', 'hoi']}  # P x D
        else:
            predictors = {k: self.output_mlps[k] for k in ['obj', 'act', 'hoi']}  # P x D

        # Final output
        logits = {k: self.repr_mlps[k](feats) @ predictors[k].t() for k in ['obj', 'act', 'hoi']}

        # Regularisation
        reg_losses = {}
        for k in ['obj', 'act', 'hoi']:
            if self.reg_coeffs[k] > 0:
                reg_losses = self.reg_coeffs[k] * self.get_reg_loss(predictors[k], self.adj_pos[k], self.adj_neg[k])
        return logits, labels, reg_losses


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
