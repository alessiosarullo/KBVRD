import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import cfg
from lib.containers import PrecomputedMinibatch
from lib.dataset.word_embeddings import WordEmbeddings
from lib.models.gcns import HicoGCN
from lib.models.misc import bce_loss, LIS
from lib.models.roi_generic_model import RoiGenericModel, Prediction, HicoDetSingleHOIsSplit


class ExtKnowledgeGenericModel(RoiGenericModel):
    @classmethod
    def get_cline_name(cls) -> str:
        raise NotImplementedError()

    def __init__(self, dataset: HicoDetSingleHOIsSplit, **kwargs):
        super().__init__(dataset, **kwargs)

        ########################################################
        # Base model
        ########################################################
        hidden_dim = 1024
        self.final_repr_dim = cfg.repr_dim
        full_dataset = self.dataset.full_dataset

        self.ho_subj_repr_mlp = nn.Sequential(*[nn.Linear(self.vis_feat_dim + full_dataset.num_objects, hidden_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=cfg.dropout),
                                                nn.Linear(hidden_dim, self.final_repr_dim),
                                                ])
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))

        self.ho_obj_repr_mlp = nn.Sequential(*[nn.Linear(self.vis_feat_dim + full_dataset.num_objects, hidden_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(p=cfg.dropout),
                                               nn.Linear(hidden_dim, self.final_repr_dim),
                                               ])
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))

        self.act_repr_mlp = nn.Sequential(*[nn.Linear(self.vis_feat_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.dropout),
                                            nn.Linear(hidden_dim, self.final_repr_dim),
                                            ])
        nn.init.xavier_normal_(self.act_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.act_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))

        num_classes = full_dataset.num_interactions if cfg.phoi else full_dataset.num_actions
        self.output_mlp = nn.Linear(self.final_repr_dim, num_classes, bias=False)
        torch.nn.init.xavier_normal_(self.output_mlp.weight, gain=1.0)

        ########################################################
        # Zero-shot
        ########################################################
        # Object-action adjacency matrix
        if cfg.oracle:
            interactions = self.dataset.full_dataset.interactions
        elif cfg.no_ext:
            interactions = self.dataset.interactions
        else:
            interactions = self.dataset.interactions
            # TODO
            raise NotImplementedError()
        oa_adj = np.zeros([self.dataset.full_dataset.num_objects, self.dataset.full_dataset.num_actions], dtype=np.float32)
        oa_adj[interactions[:, 1], interactions[:, 0]] = 1
        oa_adj[:, 0] = 0
        self.oa_adj = torch.from_numpy(oa_adj)

        word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        obj_wembs = word_embs.get_embeddings(dataset.full_dataset.objects, retry='avg')
        act_wembs = word_embs.get_embeddings(dataset.full_dataset.actions, retry='avg')
        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_wembs), requires_grad=False)
        self.act_word_embs = nn.Parameter(torch.from_numpy(act_wembs), requires_grad=False)
        self.act_emb_sim = nn.Parameter(self.act_word_embs @ self.act_word_embs.t(), requires_grad=False)

        self.zs_enabled = (cfg.seenf >= 0)
        if self.zs_enabled:
            print('Zero-shot enabled.')
            seen_act_inds = dataset.active_actions
            unseen_act_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_actions)) - set(seen_act_inds.tolist())))
            self.seen_act_inds = nn.Parameter(torch.tensor(seen_act_inds), requires_grad=False)
            self.unseen_act_inds = nn.Parameter(torch.tensor(unseen_act_inds), requires_grad=False)

            if cfg.asl:
                self.obj_act_feasibility = nn.Parameter(self.oa_adj, requires_grad=False)

    def write_soft_labels(self, vis_output: PrecomputedMinibatch):
        action_labels = vis_output.action_labels
        act_sims = self.act_emb_sim[:, self.unseen_act_inds].clamp(min=0)
        if cfg.lis:
            act_sims = LIS(act_sims, w=18, k=7)
        act_sim = action_labels @ act_sims / action_labels.sum(dim=1, keepdim=True).clamp(min=1)
        obj_labels = vis_output.box_labels[vis_output.ho_infos_np[:, 2]]
        fg_ho_pair = (obj_labels >= 0)
        unseen_action_labels = act_sim.new_zeros(act_sim.shape)
        unseen_action_labels[fg_ho_pair, :] = act_sim[fg_ho_pair, :] * self.obj_act_feasibility[:, self.unseen_act_inds][obj_labels[fg_ho_pair], :]
        action_labels[:, self.unseen_act_inds] = unseen_action_labels.detach()
        return action_labels

    def get_visual_representation(self, vis_output: PrecomputedMinibatch):
        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        hoi_infos = vis_output.ho_infos
        union_boxes_feats = vis_output.ho_union_boxes_feats

        ho_subj_feats = torch.cat([box_feats[hoi_infos[:, 1], :], boxes_ext[hoi_infos[:, 1], 5:]], dim=1)
        ho_obj_feats = torch.cat([box_feats[hoi_infos[:, 2], :], boxes_ext[hoi_infos[:, 2], 5:]], dim=1)

        ho_subj_repr = self.ho_subj_repr_mlp(ho_subj_feats)
        ho_obj_repr = self.ho_obj_repr_mlp(ho_obj_feats)
        act_repr = self.act_repr_mlp(union_boxes_feats)

        hoi_act_repr = ho_subj_repr + ho_obj_repr + act_repr
        return hoi_act_repr

    def _get_losses(self, vis_output: PrecomputedMinibatch, outputs):
        raise NotImplementedError

    def _finalize_prediction(self, prediction: Prediction, vis_output: PrecomputedMinibatch, outputs):
        action_output = outputs[0]
        assert not cfg.phoi
        assert action_output.shape[1] == self.dataset.full_dataset.num_actions
        prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()

    def _forward(self, vis_output: PrecomputedMinibatch, **kwargs):
        raise NotImplementedError()


class BaseModel(ExtKnowledgeGenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'base'

    def __init__(self, dataset: HicoDetSingleHOIsSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def _get_losses(self, vis_output: PrecomputedMinibatch, outputs):
        logits, labels = outputs

        if self.zs_enabled:
            assert cfg.asl > 0
            seen, unseen = self.seen_act_inds, self.unseen_act_inds
            if not cfg.train_null:
                seen = seen[1:]
            losses = {'act_loss': bce_loss(logits[:, seen], labels[:, seen], pos_weights=self.csp_weights),
                      'act_loss_unseen': cfg.asl * bce_loss(logits[:, unseen], labels[:, unseen])}
        else:
            losses = {'act_loss': bce_loss(logits, labels, pos_weights=self.csp_weights)}
        return losses

    def _forward(self, vis_output: PrecomputedMinibatch, **kwargs):
        hoi_act_repr = self.get_visual_representation(vis_output)
        action_logits = self.output_mlp(hoi_act_repr)
        action_labels = vis_output.action_labels
        if action_labels is not None and self.zs_enabled:
            assert cfg.asl > 0
            action_labels = self.write_soft_labels(vis_output)
        return action_logits, action_labels


class GCModel(ExtKnowledgeGenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'gc'

    def __init__(self, dataset: HicoDetSingleHOIsSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.repr_dim = 1024
        gcemb_dim = cfg.gcrdim

        # latent_dim = cfg.gcldim
        # hidden_dim = (latent_dim + self.repr_dim) // 2
        # self.emb_to_predictor = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
        #                                       nn.ReLU(inplace=True),
        #                                       nn.Dropout(p=cfg.dropout),
        #                                       nn.Linear(hidden_dim, self.repr_dim))
        # gc_dims = ((gcemb_dim + latent_dim) // 2, latent_dim)

        latent_dim = 200
        self.emb_to_predictor = nn.Sequential(nn.Linear(latent_dim, 600),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(p=cfg.dropout),
                                              nn.Linear(600, 800),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(p=cfg.dropout),
                                              nn.Linear(800, self.repr_dim),
                                              )
        gc_dims = (gcemb_dim // 2, latent_dim)

        self.gcn = HicoGCN(dataset, oa_adj=self.oa_adj, input_dim=gcemb_dim, gc_dims=gc_dims)

        if cfg.apr > 0:
            self.vv_adj = nn.Parameter((self.oa_adj.t() @ self.oa_adj).clamp(max=1).byte(), requires_grad=False)
            assert (self.vv_adj.diag()[1:] == 1).all()

    def _get_losses(self, vis_output: PrecomputedMinibatch, outputs):
        gc_logits, dir_logits, labels, reg_loss = outputs

        losses = {}
        if self.zs_enabled:
            seen, unseen = self.seen_act_inds, self.unseen_act_inds
            soft_label_loss_c = cfg.asl

            if not cfg.train_null:
                seen = seen[1:]

            losses['act_loss'] = bce_loss(dir_logits[:, seen], labels[:, seen], pos_weights=self.csp_weights) + \
                                 bce_loss(gc_logits[:, seen], labels[:, seen], pos_weights=self.csp_weights)
            # losses['act_loss'] = bce_loss(gc_logits[:, seen], labels[:, seen], pos_weights=self.csp_weights)
            if soft_label_loss_c > 0:
                losses['act_loss_unseen'] = soft_label_loss_c * bce_loss(gc_logits[:, unseen], labels[:, unseen])
        else:
            if not cfg.train_null:
                labels = labels[:, 1:]
                dir_logits = dir_logits[:, 1:]
            losses['act_loss'] = bce_loss(dir_logits, labels, pos_weights=self.csp_weights)

        if reg_loss is not None and (cfg.grg == 0 or vis_output.epoch > cfg.grg):
            losses['act_reg_loss'] = reg_loss
        return losses

    def _finalize_prediction(self, prediction: Prediction, vis_output: PrecomputedMinibatch, outputs):
        gc_logits, dir_logits, labels, reg_loss = outputs
        logits = dir_logits
        if self.zs_enabled:
            logits[:, self.unseen_act_inds] = gc_logits[:, self.unseen_act_inds]
        prediction.action_scores = torch.sigmoid(logits).cpu().numpy()

    def _forward(self, vis_output: PrecomputedMinibatch, **kwargs):
        vrepr = self.get_visual_representation(vis_output)
        dir_act_logits = self.output_mlp(vrepr)
        _, act_class_embs = self.gcn()  # P x E
        # act_predictors = act_class_embs  # P x D
        act_predictors = self.emb_to_predictor(act_class_embs)  # P x D
        gcn_act_logits = vrepr @ act_predictors.t()

        action_labels = vis_output.action_labels
        if action_labels is not None and self.zs_enabled:
            if cfg.asl > 0:
                action_labels = self.write_soft_labels(vis_output)

        reg_loss = None
        if cfg.apr > 0:
            adj = self.vv_adj
            seen = self.seen_act_inds
            unseen = self.unseen_act_inds

            # Detach seen classes predictors
            all_trainable_predictors = act_predictors
            predictors_seen = act_predictors[seen, :].detach()
            predictors_unseen = act_predictors[unseen, :]
            predictors = torch.cat([predictors_seen, predictors_unseen], dim=0)[torch.sort(torch.cat([seen, unseen]))[1]]
            assert (all_trainable_predictors[seen] == predictors[seen]).all() and (all_trainable_predictors[unseen] == predictors[unseen]).all()

            predictors_norm = F.normalize(predictors, dim=1)
            predictors_sim = predictors_norm @ predictors_norm.t()
            null = ~adj.any(dim=1)
            arange = torch.arange(predictors_sim.shape[0])

            predictors_sim_diff = predictors_sim.unsqueeze(dim=2) - predictors_sim.unsqueeze(dim=1)
            reg_loss_mat = (cfg.grm - predictors_sim_diff).clamp(min=0)
            reg_loss_mat[~adj.unsqueeze(dim=2).expand_as(reg_loss_mat)] = 0
            reg_loss_mat[adj.unsqueeze(dim=1).expand_as(reg_loss_mat)] = 0
            reg_loss_mat[arange, arange, :] = 0
            reg_loss_mat[arange, :, arange] = 0
            reg_loss_mat[:, arange, arange] = 0
            reg_loss_mat[null, :, :] = 0
            reg_loss_mat[:, null, :] = 0
            reg_loss_mat[:, :, null] = 0

            reg_loss_mat = reg_loss_mat[unseen, :, :]
            reg_loss = reg_loss_mat.sum() / (reg_loss_mat != 0).sum().item()

        return gcn_act_logits, dir_act_logits, action_labels, reg_loss  # order is important!
