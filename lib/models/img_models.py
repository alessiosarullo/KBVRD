from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import cfg
from lib.containers import Prediction
from lib.dataset.hico import HicoSplit
from lib.dataset.utils import interactions_to_mat, get_noun_verb_adj_mat
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.abstract_model import AbstractModel
from lib.models.gcns import HicoGCN, WEmbHicoGCN
from lib.models.misc import bce_loss


class SKZSMultiModel(AbstractModel):
    @classmethod
    def get_cline_name(cls):
        return 'skzs'

    def __init__(self, dataset: HicoSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.dataset = dataset
        self.repr_dim = cfg.repr_dim
        self.loss_coeffs = {'obj': cfg.olc, 'act': cfg.alc}
        self.soft_label_loss_coeffs = {'obj': cfg.osl, 'act': cfg.asl}
        self.reg_coeffs = {'obj': cfg.opr, 'act': cfg.apr}
        self.soft_labels_enabled = any([v > 0 for v in self.soft_label_loss_coeffs.values()])

        # Word embeddings + similarity matrices
        word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        obj_wembs = word_embs.get_embeddings(dataset.full_dataset.objects, retry='avg')
        act_wembs = word_embs.get_embeddings(dataset.full_dataset.actions, retry='avg')
        self.word_embs = nn.ParameterDict({'obj': nn.Parameter(torch.from_numpy(obj_wembs), requires_grad=False),
                                           'act': nn.Parameter(torch.from_numpy(act_wembs), requires_grad=False)
                                           })
        self.obj_emb_sim = nn.Parameter(self.word_embs['obj'] @ self.word_embs['obj'].t(), requires_grad=False)
        self.act_emb_sim = nn.Parameter(self.word_embs['act'] @ self.word_embs['act'].t(), requires_grad=False)
        self.act_obj_emb_sim = nn.Parameter(self.word_embs['act'] @ self.word_embs['obj'].t(), requires_grad=False)

        # Object-action adjacency matrix
        if cfg.oracle:
            interactions = self.dataset.full_dataset.interactions
        elif cfg.no_ext:
            interactions = self.dataset.interactions
        else:
            interactions = self.dataset.interactions
            # TODO
        oa_adj = np.zeros([self.dataset.full_dataset.num_objects, self.dataset.full_dataset.num_actions], dtype=np.float32)
        oa_adj[interactions[:, 1], interactions[:, 0]] = 1
        oa_adj = torch.from_numpy(oa_adj)

        # Base model
        self.repr_mlps = nn.ModuleDict()
        for k in ['obj', 'act']:
            self.repr_mlps[k] = nn.Sequential(*[nn.Linear(self.dataset.precomputed_visual_feat_dim, 1024),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=cfg.dropout),
                                                nn.Linear(1024, self.repr_dim),
                                                ])
            nn.init.xavier_normal_(self.repr_mlps[k][0].weight, gain=torch.nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.repr_mlps[k][3].weight, gain=torch.nn.init.calculate_gain('linear'))

        # Predictors
        self.use_gc = not cfg.no_gc
        self.use_wemb = not cfg.no_wemb
        self.linear_predictors = nn.ParameterDict()
        for k, d in [('obj', dataset.full_dataset.num_objects),
                     ('act', dataset.full_dataset.num_actions),
                     ]:
            self.linear_predictors[k] = nn.Parameter(torch.empty(d, self.repr_dim), requires_grad=True)
            torch.nn.init.xavier_normal_(self.linear_predictors[k], gain=1.0)
        if self.use_gc:
            gcemb_dim = cfg.gcrdim
            latent_dim = cfg.gcldim
            hidden_dim = (latent_dim + self.repr_dim) // 2
            self.predictor_mlps = nn.ModuleDict({k: nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                                                  nn.ReLU(inplace=True),
                                                                  nn.Dropout(p=cfg.gcdropout),
                                                                  nn.Linear(hidden_dim, self.repr_dim),
                                                                  ) for k in ['obj', 'act']})
            gc_dims = ((gcemb_dim + latent_dim) // 2, latent_dim)

            if cfg.gcwemb:
                self.gcn = WEmbHicoGCN(dataset, input_dim=gcemb_dim, gc_dims=gc_dims)
            else:
                self.gcn = HicoGCN(dataset, oa_adj=oa_adj, input_dim=gcemb_dim, gc_dims=gc_dims)
        if self.use_wemb:
            self.wemb_mlps = nn.ModuleDict()
            for k in ['obj', 'act']:
                self.wemb_mlps[k] = nn.Sequential(*[nn.Linear(word_embs.dim, 600),
                                                    nn.ReLU(inplace=True),
                                                    nn.Linear(600, self.repr_dim),
                                                    ])
                nn.init.xavier_normal_(self.wemb_mlps[k][0].weight, gain=torch.nn.init.calculate_gain('relu'))
                nn.init.xavier_normal_(self.wemb_mlps[k][2].weight, gain=torch.nn.init.calculate_gain('linear'))

        # Regardless of GCN connectivity, now we do not want the null action to be connected.
        oa_adj[:, 0] = 0

        # Seen/unseen indices
        self.zs_enabled = (cfg.seenf >= 0)
        if self.zs_enabled:
            print('Zero-shot enabled.')
            self.seen_inds = {}
            self.unseen_inds = {}

            seen_obj_inds = dataset.active_objects
            unseen_obj_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_objects)) - set(seen_obj_inds.tolist())))
            self.seen_inds['obj'] = nn.Parameter(torch.tensor(seen_obj_inds), requires_grad=False)
            self.unseen_inds['obj'] = nn.Parameter(torch.tensor(unseen_obj_inds), requires_grad=False)

            seen_act_inds = dataset.active_actions
            unseen_act_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_actions)) - set(seen_act_inds.tolist())))
            self.seen_inds['act'] = nn.Parameter(torch.tensor(seen_act_inds), requires_grad=False)
            self.unseen_inds['act'] = nn.Parameter(torch.tensor(unseen_act_inds), requires_grad=False)

            if self.soft_labels_enabled:
                self.obj_act_feasibility = nn.Parameter(oa_adj, requires_grad=False)

        # Regularisation
        self.adj = nn.ParameterDict()
        if not cfg.sgr:
            if self.reg_coeffs['obj'] > 0:
                self.adj['obj'] = nn.Parameter((oa_adj @ oa_adj.t()).clamp(max=1).byte(), requires_grad=False)
            if self.reg_coeffs['act'] > 0:
                self.adj['act'] = nn.Parameter((oa_adj.t() @ oa_adj).clamp(max=1).byte(), requires_grad=False)
        else:
            self.reg_margin = nn.Parameter(self.act_emb_sim.unsqueeze(dim=2) - self.act_emb_sim.unsqueeze(dim=1), requires_grad=False)

        self._csp_weights = {}
        if any([cfg.ocs, cfg.acs]):
            raise NotImplementedError('Check that the use of HICO interactions is ok for ZS.')
            if not cfg.train_null:
                null_interactions = set(np.flatnonzero(self.dataset.full_dataset.interactions[:, 0] == 0).tolist())
                interactions = np.array(sorted(set(self.dataset.active_interactions.tolist()) - null_interactions))
                actions = np.array(sorted(set(self.dataset.active_actions.tolist()) - {0}))
            else:
                interactions = self.dataset.active_interactions
                actions = self.dataset.active_actions
            all_interactions_hist = np.sum(self.dataset.labels, axis=0)  # Unseen interactions have a value of 0 anyway, but this preserves indices.

            hist = {}
            if cfg.ocs:
                hist['obj'] = all_interactions_hist @ self.dataset.full_dataset.interaction_to_object_mat[:, self.dataset.active_objects]
            if cfg.acs:
                hist['act'] = all_interactions_hist @ self.dataset.full_dataset.interaction_to_action_mat[:, actions]

            csp_weights = {}
            for k, h in hist.items():
                cost_matrix = np.maximum(1, np.log2(h[None, :] / h[:, None])) * (1 - np.eye(h.size))
                tot = h.sum()
                csp_weights[k] = cost_matrix @ h / (tot - h)
            self._csp_weights = nn.ParameterDict({k: nn.Parameter(torch.from_numpy(v).float(), requires_grad=False) for k, v in csp_weights.items()})
        self.csp_weights = {k: self._csp_weights[k] if k in self._csp_weights else None for k in ['obj', 'act']}

    def get_soft_labels(self, labels):
        assert cfg.osl or cfg.asl
        batch_size = labels.shape[0]

        inter_mat = interactions_to_mat(labels, hico=self.dataset.full_dataset)  # N x I -> N x O x P
        ext_inter_mat = inter_mat

        obj_labels = (labels @ torch.from_numpy(self.dataset.full_dataset.interaction_to_object_mat).to(labels)).clamp(max=1)
        if cfg.osl:
            obj_emb_sim = self.obj_emb_sim.clamp(min=0)
            similar_obj_per_act = torch.bmm(inter_mat.transpose(1, 2), obj_emb_sim.unsqueeze(dim=0).expand(batch_size, -1, -1)).transpose(2, 1)
            similar_obj_per_act = similar_obj_per_act / inter_mat.sum(dim=1, keepdim=True).clamp(min=1)
            similar_obj_per_act = similar_obj_per_act * self.obj_act_feasibility.unsqueeze(dim=0).expand(batch_size, -1, -1)
            ext_inter_mat += similar_obj_per_act

            obj_labels[:, self.unseen_inds['obj']] = similar_obj_per_act.max(dim=2)[0][:, self.unseen_inds['obj']]
        obj_labels = obj_labels.detach()

        act_labels = (labels @ torch.from_numpy(self.dataset.full_dataset.interaction_to_action_mat).to(labels)).clamp(max=1)
        if cfg.asl:
            if cfg.slpure:
                feasible_acts_per_obj = ext_inter_mat.max(dim=2)[0] @ self.obj_act_feasibility
                act_labels[:, self.unseen_inds['act']] = feasible_acts_per_obj.clamp(max=1)[:, self.unseen_inds['act']]  # group across objects
            else:
                act_emb_sims = self.act_emb_sim.clamp(min=0)
                similar_acts_per_obj = torch.bmm(ext_inter_mat, act_emb_sims.unsqueeze(dim=0).expand(batch_size, -1, -1))
                similar_acts_per_obj = similar_acts_per_obj / ext_inter_mat.sum(dim=2, keepdim=True).clamp(min=1)
                feasible_similar_acts_per_obj = similar_acts_per_obj * self.obj_act_feasibility.unsqueeze(dim=0).expand(batch_size, -1, -1)
                act_labels[:, self.unseen_inds['act']] = feasible_similar_acts_per_obj.max(dim=1)[0][:,
                                                         self.unseen_inds['act']]  # group across objects
        act_labels = act_labels.detach()

        return obj_labels, act_labels

    def get_reg_loss(self, predictors, branch):
        seen = self.seen_inds[branch]
        unseen = self.unseen_inds[branch]

        # Detach seen classes predictors
        if not cfg.grseen:
            all_trainable_predictors = predictors
            predictors_seen = predictors[seen, :].detach()
            predictors_unseen = predictors[unseen, :]
            predictors = torch.cat([predictors_seen, predictors_unseen], dim=0)[torch.sort(torch.cat([seen, unseen]))[1]]
            assert (all_trainable_predictors[seen] == predictors[seen]).all() and (all_trainable_predictors[unseen] == predictors[unseen]).all()

        predictors_norm = F.normalize(predictors, dim=1)
        predictors_sim = predictors_norm @ predictors_norm.t()
        predictors_sim_diff = predictors_sim.unsqueeze(dim=2) - predictors_sim.unsqueeze(dim=1)  # neighbours along axis 1, non-neighbours along 2
        arange = torch.arange(predictors_sim.shape[0])

        if cfg.sgr:
            if branch != 'act':
                raise NotImplementedError()
            reg_loss_mat = (self.reg_margin - predictors_sim_diff).clamp(min=0)
        else:
            adj = self.adj[branch]
            null = ~adj.any(dim=1)

            reg_loss_mat = (cfg.grm - predictors_sim_diff).clamp(min=0)
            reg_loss_mat[~adj.unsqueeze(dim=2).expand_as(reg_loss_mat)] = 0
            reg_loss_mat[adj.unsqueeze(dim=1).expand_as(reg_loss_mat)] = 0
            reg_loss_mat[null, :, :] = 0
            reg_loss_mat[:, null, :] = 0
            reg_loss_mat[:, :, null] = 0

        reg_loss_mat[arange, arange, :] = 0
        reg_loss_mat[arange, :, arange] = 0
        reg_loss_mat[:, arange, arange] = 0
        reg_loss_mat = reg_loss_mat[unseen, :, :]
        reg_loss = reg_loss_mat.sum() / (reg_loss_mat != 0).sum().item()
        return reg_loss

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):

            feats, orig_labels, other = x
            if not inference:
                epoch, iter = other
            all_instance_reprs, all_linear_predictors, all_knowl_predictors, all_labels = self._forward(feats, orig_labels)
            all_logits = {k: all_instance_reprs[k] @ all_linear_predictors[k].t() for k in ['obj', 'act']}
            if self.use_gc or self.use_wemb:
                all_knowl_logits = {k: all_instance_reprs[k] @ all_knowl_predictors[k].t() for k in ['obj', 'act']}
            else:
                all_knowl_logits = {}

            if not inference:
                all_labels = {k: v for k, v in all_labels.items()}
                losses = {}
                for k in ['obj', 'act']:
                    logits = all_logits[k]
                    labels = all_labels[k]
                    loss_c = self.loss_coeffs[k]
                    if loss_c == 0:
                        continue

                    # Loss
                    if self.zs_enabled:
                        seen, unseen = self.seen_inds[k], self.unseen_inds[k]
                        soft_label_loss_c = self.soft_label_loss_coeffs[k]
                        if not cfg.train_null:
                            if k == 'act':
                                seen = seen[1:]
                        if len(all_knowl_logits) > 0:
                            losses[f'{k}_loss'] = loss_c * bce_loss(logits[:, seen], labels[:, seen], pos_weights=self.csp_weights[k]) + \
                                                  loss_c * bce_loss(all_knowl_logits[k][:, seen], labels[:, seen], pos_weights=self.csp_weights[k])
                            if soft_label_loss_c > 0:
                                losses[f'{k}_loss_unseen'] = loss_c * soft_label_loss_c * bce_loss(all_knowl_logits[k][:, unseen], labels[:, unseen])
                        else:
                            losses[f'{k}_loss'] = loss_c * bce_loss(logits[:, seen], labels[:, seen], pos_weights=self.csp_weights[k])
                            if soft_label_loss_c > 0:
                                losses[f'{k}_loss_unseen'] = loss_c * soft_label_loss_c * bce_loss(logits[:, unseen], labels[:, unseen])
                    else:
                        if not cfg.train_null:
                            if k == 'act':
                                labels = labels[:, 1:]
                                logits = logits[:, 1:]
                        losses[f'{k}_loss'] = loss_c * bce_loss(logits, labels, pos_weights=self.csp_weights[k])

                    # Regularisation loss
                    if self.reg_coeffs[k] > 0:
                        if cfg.grg == 0 or epoch > cfg.grg:
                            reg_loss = self.get_reg_loss(all_knowl_predictors[k], branch=k)
                            losses[f'{k}_reg_loss'] = self.reg_coeffs[k] * reg_loss
                return losses
            else:
                if self.zs_enabled and len(all_knowl_logits) > 0:
                    for k in all_logits.keys():
                        unseen = self.unseen_inds[k]
                        all_logits[k][:, unseen] = all_knowl_logits[k][:, unseen]
                prediction = Prediction()
                interactions = self.dataset.full_dataset.interactions
                obj_scores = torch.sigmoid(all_logits['obj']).cpu().numpy() ** cfg.osc
                act_scores = torch.sigmoid(all_logits['act']).cpu().numpy() ** cfg.asc
                prediction.hoi_scores = obj_scores[:, interactions[:, 1]] * act_scores[:, interactions[:, 0]]
                return prediction

    def _forward(self, feats, labels):
        # Predictors
        linear_predictors = {k: self.linear_predictors[k] for k in ['obj', 'act']}  # P x D
        if self.use_gc:
            obj_class_embs, act_class_embs = self.gcn()  # P x E
            class_embs = {'obj': obj_class_embs, 'act': act_class_embs}
            predictors = {k: self.predictor_mlps[k](class_embs[k]) for k in ['obj', 'act']}  # P x D
            if self.use_wemb:
                branches = ['obj']
                if cfg.wemb_a:
                    branches.append('act')
                wemb_predictors = {k: self.wemb_mlps[k](self.word_embs[k]) for k in branches}
                predictors = {k: v + wemb_predictors.get(k, 0) for k, v in predictors.items()}
        elif self.use_wemb:
            predictors = {k: self.wemb_mlps[k](self.word_embs[k]) for k in ['obj', 'act']}
        else:
            predictors = linear_predictors

        # Instance representation
        instance_repr = {k: self.repr_mlps[k](feats) for k in ['obj', 'act']}

        # Labels
        if labels is not None:
            obj_labels = (labels @ torch.from_numpy(self.dataset.full_dataset.interaction_to_object_mat).to(labels)).clamp(max=1).detach()
            act_labels = (labels @ torch.from_numpy(self.dataset.full_dataset.interaction_to_action_mat).to(labels)).clamp(max=1).detach()
            if self.soft_labels_enabled:
                obj_labels, act_labels = self.get_soft_labels(labels)
        else:
            obj_labels = act_labels = None
        labels = {'obj': obj_labels, 'act': act_labels}
        return instance_repr, linear_predictors, predictors, labels


class WEmbModel(AbstractModel):
    @classmethod
    def get_cline_name(cls):
        return 'wemb'

    def __init__(self, dataset: HicoSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.dataset = dataset
        self.repr_dim = cfg.repr_dim
        self.loss_coeffs = {'obj': cfg.olc, 'act': cfg.alc}

        # Word embeddings + similarity matrices
        word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        obj_wembs = word_embs.get_embeddings(dataset.full_dataset.objects, retry='avg')
        act_wembs = word_embs.get_embeddings(dataset.full_dataset.actions, retry='avg')
        self.word_embs = nn.ParameterDict({'obj': nn.Parameter(torch.from_numpy(obj_wembs), requires_grad=False),
                                           'act': nn.Parameter(torch.from_numpy(act_wembs), requires_grad=False)
                                           })

        # Seen/unseen indices
        self.zs_enabled = (cfg.seenf >= 0)
        assert self.zs_enabled
        print('Zero-shot enabled.')
        self.seen_inds = {}
        self.unseen_inds = {}

        seen_obj_inds = dataset.active_objects
        unseen_obj_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_objects)) - set(seen_obj_inds.tolist())))
        self.seen_inds['obj'] = nn.Parameter(torch.tensor(seen_obj_inds), requires_grad=False)
        self.unseen_inds['obj'] = nn.Parameter(torch.tensor(unseen_obj_inds), requires_grad=False)

        seen_act_inds = dataset.active_actions
        unseen_act_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_actions)) - set(seen_act_inds.tolist())))
        self.seen_inds['act'] = nn.Parameter(torch.tensor(seen_act_inds), requires_grad=False)
        self.unseen_inds['act'] = nn.Parameter(torch.tensor(unseen_act_inds), requires_grad=False)

        # Base model
        self.repr_mlps = nn.ModuleDict()
        self.wemb_mlps = nn.ModuleDict()
        for k, d in [('obj', dataset.full_dataset.num_objects),
                     ('act', dataset.full_dataset.num_actions),
                     ]:
            self.repr_mlps[k] = nn.Sequential(*[nn.Linear(self.dataset.precomputed_visual_feat_dim, 1024),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=cfg.dropout),
                                                nn.Linear(1024, self.repr_dim),
                                                ])
            nn.init.xavier_normal_(self.repr_mlps[k][0].weight, gain=torch.nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.repr_mlps[k][3].weight, gain=torch.nn.init.calculate_gain('linear'))
            self.wemb_mlps[k] = nn.Sequential(*[nn.Linear(word_embs.dim, 600),
                                                nn.ReLU(inplace=True),
                                                nn.Linear(600, self.repr_dim),
                                                ])
            nn.init.xavier_normal_(self.wemb_mlps[k][0].weight, gain=torch.nn.init.calculate_gain('relu'))
            nn.init.xavier_normal_(self.wemb_mlps[k][2].weight, gain=torch.nn.init.calculate_gain('linear'))

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            feats, orig_labels, _ = x
            all_logits, all_labels = self._forward(feats, orig_labels)

            if not inference:
                all_labels = {k: v for k, v in all_labels.items()}
                losses = {}
                for k in ['obj', 'act', ]:
                    logits = all_logits[k]
                    labels = all_labels[k]
                    loss_c = self.loss_coeffs[k]
                    if loss_c == 0:
                        continue

                    seen, unseen = self.seen_inds[k], self.unseen_inds[k]
                    if not cfg.train_null and k == 'act':
                        seen = seen[1:]
                    losses[f'{k}_loss'] = loss_c * bce_loss(logits[:, seen], labels[:, seen])
                return losses
            else:
                prediction = Prediction()
                interactions = self.dataset.full_dataset.interactions
                obj_scores = torch.sigmoid(all_logits['obj']).cpu().numpy() ** cfg.osc
                act_scores = torch.sigmoid(all_logits['act']).cpu().numpy() ** cfg.asc
                prediction.hoi_scores = obj_scores[:, interactions[:, 1]] * act_scores[:, interactions[:, 0]]
                return prediction

    def _forward(self, feats, labels):
        # Labels
        if labels is not None:
            obj_labels = (labels @ torch.from_numpy(self.dataset.full_dataset.interaction_to_object_mat).to(labels)).clamp(max=1).detach()
            act_labels = (labels @ torch.from_numpy(self.dataset.full_dataset.interaction_to_action_mat).to(labels)).clamp(max=1).detach()
        else:
            obj_labels = act_labels = None
        labels = {'obj': obj_labels, 'act': act_labels}

        # Logits
        logits = {k: self.repr_mlps[k](feats) @ self.wemb_mlps[k](self.word_embs[k]).t() for k in ['obj', 'act']}
        return logits, labels
