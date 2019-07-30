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
from lib.models.branches import get_noun_verb_adj_mat, CheatGCNBranch
from lib.models.containers import Prediction
from lib.models.misc import bce_loss, LIS


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

            feats, labels = x
            output = self._forward(feats, labels)

            if not inference:
                zero_labels = (labels == 0)
                labels.clamp_(min=0)
                loss_mat = bce_loss(output, labels, reduce=False)
                if cfg.hico_lhard:
                    loss_mat[zero_labels] = 0
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


class HicoExtKnowledgeGenericModel(AbstractModel):
    def __init__(self, dataset: HicoSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        assert cfg.hico
        self.dataset = dataset

        self.nv_adj = get_noun_verb_adj_mat(dataset=dataset, iso_null=True)

        word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        obj_wembs = word_embs.get_embeddings(dataset.full_dataset.objects, retry='avg')
        pred_wembs = word_embs.get_embeddings(dataset.full_dataset.predicates, retry='avg')
        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_wembs), requires_grad=False)
        self.obj_emb_sim = nn.Parameter(self.obj_word_embs @ self.obj_word_embs.t(), requires_grad=False)
        self.pred_word_embs = nn.Parameter(torch.from_numpy(pred_wembs), requires_grad=False)
        self.pred_emb_sim = nn.Parameter(self.pred_word_embs @ self.pred_word_embs.t(), requires_grad=False)

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

    def interactions_to_actions(self, hois):
        hico = self.dataset.full_dataset
        i_to_a_mat = np.zeros((hico.num_interactions, hico.num_predicates))
        i_to_a_mat[np.arange(hico.num_interactions), hico.interactions[:, 0]] = 1
        i_to_a_mat = torch.from_numpy(i_to_a_mat).to(hois)
        actions = (hois @ i_to_a_mat).clamp(max=1)
        return actions

    def interactions_to_objects(self, hois):
        hico = self.dataset.full_dataset
        i_to_o_mat = np.zeros((hico.num_interactions, hico.num_object_classes))
        i_to_o_mat[np.arange(hico.num_interactions), hico.interactions[:, 1]] = 1
        i_to_o_mat = torch.from_numpy(i_to_o_mat).to(hois)
        objects = (hois @ i_to_o_mat).clamp(max=1)
        return objects

    def interactions_to_mat(self, hois):
        hico = self.dataset.full_dataset
        hois_np = hois.detach().cpu().numpy()
        all_hois = np.stack(np.where(hois_np > 0), axis=1)
        all_interactions = np.concatenate([all_hois[:, :1], hico.interactions[all_hois[:, 1], :]], axis=1)
        inter_mat = np.zeros((hois.shape[0], hico.num_object_classes, hico.num_predicates))
        inter_mat[all_interactions[:, 0], all_interactions[:, 1], all_interactions[:, 2]] = 1
        inter_mat = torch.from_numpy(inter_mat).to(hois)
        return inter_mat

    def get_soft_labels(self, labels):
        actions = self.interactions_to_actions(labels)
        objects = self.interactions_to_objects(labels)
        inter_mat = self.interactions_to_mat(labels.clamp(min=0))

        act_sim_per_obj = torch.bmm(inter_mat, self.pred_emb_sim.unsqueeze(dim=0)) / inter_mat.sum(dim=2, keepdim=True).clamp(min=1)

        interactions = self.dataset.full_dataset.interactions
        unseen_labels = act_sim_per_obj[:, interactions[:, 1], interactions[:, 0]][:, self.unseen_hoi_inds]
        return unseen_labels.detach()

    def forward(self, x: List[torch.Tensor], inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):

            feats, labels = x
            logits, labels, reg_loss = self._forward(feats, labels)

            if not inference:
                labels.clamp_(min=0)
                if cfg.softl > 0:
                    # if cfg.nullzs:
                    #     unseen_action_labels *= (1 - action_labels[:, :1])  # cannot be anything else if it is a positive (i.e., from GT) null
                    losses = {'hoi_loss': bce_loss(logits[:, self.seen_hoi_inds], labels[:, self.seen_hoi_inds]),
                              'hoi_loss_unseen': cfg.softl * bce_loss(logits[:, self.unseen_hoi_inds], labels[:, self.unseen_hoi_inds])}
                else:
                    losses = {'hoi_loss': bce_loss(logits, labels)}
                if reg_loss is not None:
                    losses['reg_loss'] = reg_loss
                return losses
            else:
                prediction = Prediction()
                prediction.hoi_scores = torch.sigmoid(logits).cpu().numpy()
                return prediction


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
        self.predictor_dim = 1024

        gcemb_dim = 1024
        if cfg.puregc:
            self.gcn = CheatGCNBranch(dataset, input_repr_dim=gcemb_dim, gc_dims=(gcemb_dim, self.predictor_dim))
        else:
            latent_dim = 200
            input_dim = self.predictor_dim
            self.emb_to_predictor = nn.Sequential(nn.Linear(latent_dim, 600),
                                                  nn.ReLU(inplace=True),
                                                  nn.Dropout(p=cfg.dropout),
                                                  nn.Linear(600, 800),
                                                  nn.ReLU(inplace=True),
                                                  nn.Dropout(p=cfg.dropout),
                                                  nn.Linear(800, input_dim),
                                                  )
            self.gcn = CheatGCNBranch(dataset, input_repr_dim=gcemb_dim, gc_dims=(gcemb_dim // 2, latent_dim))

        if cfg.greg > 0:
            raise NotImplementedError
            self.vv_adj = nn.Parameter((self.nv_adj.t() @ self.nv_adj).clamp(max=1).byte(), requires_grad=False)
            assert (self.vv_adj.diag()[1:] == 1).all()

    def _forward(self, feats, labels):
        vrepr = self.base_model._forward(feats, labels, return_repr=True)

        hico = self.dataset.full_dataset
        obj_class_embs, act_class_embs = self.gcn()  # P x E
        hoi_class_embs = (obj_class_embs[hico.interactions[:, 1]] + act_class_embs[hico.interactions[:, 0]]) / 2
        hoi_predictors = self.emb_to_predictor(hoi_class_embs)  # P x D
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
            neigh_mask[self.vv_adj] = 1
            argmin_neigh_sim = (hoi_predictors_sim * neigh_mask.detach()).argmin(dim=1)
            min_neigh_sim = hoi_predictors_sim[arange, argmin_neigh_sim]

            non_neigh_mask = torch.full_like(hoi_predictors_sim, -np.inf)
            non_neigh_mask[~self.vv_adj] = 1
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