import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from config import cfg
from lib.dataset.utils import get_noun_verb_adj_mat
from lib.dataset.word_embeddings import WordEmbeddings
from lib.containers import PrecomputedMinibatch
from lib.models.gcns import HicoGCN
from lib.models.misc import bce_loss, LIS
from lib.models.roi_generic_model import GenericModel, Prediction, HicoDetSingleHOIsSplit


class BaseModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'base'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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

    def _forward(self, vis_output: PrecomputedMinibatch, step=None, epoch=None, return_repr=False):
        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        hoi_infos = vis_output.ho_infos
        union_boxes_feats = vis_output.ho_union_boxes_feats

        subj_ho_feats = torch.cat([box_feats[hoi_infos[:, 1], :], boxes_ext[hoi_infos[:, 1], 5:]], dim=1)
        obj_ho_feats = torch.cat([box_feats[hoi_infos[:, 2], :], boxes_ext[hoi_infos[:, 2], 5:]], dim=1)

        ho_subj_repr = self.ho_subj_repr_mlp(subj_ho_feats)
        ho_obj_repr = self.ho_obj_repr_mlp(obj_ho_feats)
        act_repr = self.act_repr_mlp(union_boxes_feats)

        hoi_act_repr = ho_subj_repr + ho_obj_repr + act_repr

        output_logits = self.output_mlp(hoi_act_repr)

        if cfg.monitor:
            self.values_to_monitor['ho_subj_repr'] = ho_subj_repr
            self.values_to_monitor['ho_obj_repr'] = ho_obj_repr
            self.values_to_monitor['act_repr'] = act_repr
            self.values_to_monitor['output_logits'] = output_logits

        if return_repr:
            return output_logits, hoi_act_repr
        else:
            return output_logits


class ExtKnowledgeGenericModel(GenericModel):
    def __init__(self, dataset: HicoDetSingleHOIsSplit, **kwargs):
        super().__init__(dataset, **kwargs)

        self.nv_adj = get_noun_verb_adj_mat(dataset=dataset, isolate_null=True)

        word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        obj_wembs = word_embs.get_embeddings(dataset.full_dataset.objects, retry='avg')
        act_wembs = word_embs.get_embeddings(dataset.full_dataset.actions, retry='avg')
        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_wembs), requires_grad=False)
        self.act_word_embs = nn.Parameter(torch.from_numpy(act_wembs), requires_grad=False)
        self.act_emb_sim = nn.Parameter(self.act_word_embs @ self.act_word_embs.t(), requires_grad=False)

        self.zs_enabled = (cfg.seenf >= 0)
        self.load_backbone = len(cfg.hoi_backbone) > 0
        if self.zs_enabled:
            print('Zero-shot enabled.')
            seen_act_inds = dataset.active_actions
            unseen_act_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_actions)) - set(seen_act_inds.tolist())))
            self.seen_act_inds = nn.Parameter(torch.tensor(seen_act_inds), requires_grad=False)
            self.unseen_act_inds = nn.Parameter(torch.tensor(unseen_act_inds), requires_grad=False)

            if self.load_backbone:
                ckpt = torch.load(cfg.hoi_backbone)
                self.pretrained_base_model = BaseModel(dataset)
                self.pretrained_base_model.load_state_dict(ckpt['state_dict'])
                self.pretrained_predictors = nn.Parameter(self.pretrained_base_model.output_mlp.weight.detach(), requires_grad=False)  # P x D
                assert len(seen_act_inds) == self.pretrained_predictors.shape[0]

            if cfg.asl:
                self.obj_act_feasibility = nn.Parameter(self.nv_adj, requires_grad=False)

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

    def _refine_output(self, vis_output, inference, outputs):
        if inference and self.load_backbone:
            action_output, action_labels, reg_loss, unseen_action_labels = outputs
            _, pretrained_vrepr = self.pretrained_base_model._forward(vis_output, return_repr=True).detach()
            pretrained_action_output = pretrained_vrepr @ self.pretrained_predictors.t()  # N x Pt
            action_output[:, self.seen_act_inds] = pretrained_action_output
            outputs = action_output, action_labels, reg_loss, unseen_action_labels
        return outputs

    def _get_losses(self, vis_output: PrecomputedMinibatch, outputs):
        raise NotImplementedError

    def _finalize_prediction(self, prediction: Prediction, vis_output: PrecomputedMinibatch, outputs):
        action_output = outputs[0]
        assert not cfg.phoi
        assert action_output.shape[1] == self.dataset.full_dataset.num_actions
        prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()


class ZSBaseModel(ExtKnowledgeGenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'zsb'

    def __init__(self, dataset: HicoDetSingleHOIsSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.base_model = BaseModel(dataset, **kwargs)
        assert self.zs_enabled and cfg.asl > 0

    def _get_losses(self, vis_output: PrecomputedMinibatch, outputs):
        logits, labels = outputs

        seen, unseen = self.seen_act_inds, self.unseen_act_inds
        if not cfg.train_null:
            seen = seen[1:]

        losses = {'act_loss': bce_loss(logits[:, seen], labels[:, seen], pos_weights=self.csp_weights),
                  'act_loss_unseen': cfg.asl * bce_loss(logits[:, unseen], labels[:, unseen])}
        return losses

    def _forward(self, vis_output: PrecomputedMinibatch, step=None, epoch=None, **kwargs):
        action_logits = self.base_model._forward(vis_output)
        action_labels = vis_output.action_labels
        if action_labels is not None:
            action_labels = self.write_soft_labels(vis_output)
        return action_logits, action_labels


class ZSGCModel(ExtKnowledgeGenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'zsgc'

    def __init__(self, dataset: HicoDetSingleHOIsSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.base_model = BaseModel(dataset, **kwargs)
        self.repr_dim = 1024

        gcemb_dim = cfg.gcrdim
        latent_dim = cfg.gcldim
        hidden_dim = (latent_dim + self.repr_dim) // 2
        self.emb_to_predictor = nn.Sequential(nn.Linear(latent_dim, hidden_dim),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(p=cfg.dropout),
                                              nn.Linear(hidden_dim, self.repr_dim))
        gc_dims = ((gcemb_dim + latent_dim) // 2, latent_dim)
        self.gcn = HicoGCN(dataset, input_dim=gcemb_dim, gc_dims=gc_dims)

        if cfg.apr > 0:
            self.vv_adj = nn.Parameter((self.nv_adj.t() @ self.nv_adj).clamp(max=1).byte(), requires_grad=False)
            assert (self.vv_adj.diag()[1:] == 1).all()

    def _get_losses(self, vis_output: PrecomputedMinibatch, outputs):
        dir_logits, gc_logits, labels, reg_loss = outputs

        losses = {}
        if self.zs_enabled:
            seen, unseen = self.seen_act_inds, self.unseen_act_inds
            soft_label_loss_c = cfg.asl

            if not cfg.train_null:
                seen = seen[1:]

            losses['act_loss'] = bce_loss(dir_logits[:, seen], labels[:, seen], pos_weights=self.csp_weights) + \
                                 bce_loss(gc_logits[:, seen], labels[:, seen], pos_weights=self.csp_weights)
            if soft_label_loss_c > 0:
                losses['act_loss_unseen'] = soft_label_loss_c * bce_loss(gc_logits[:, unseen], labels[:, unseen])
        else:
            if not cfg.train_null:
                labels = labels[:, 1:]
                dir_logits = dir_logits[:, 1:]
            losses['act_loss'] = bce_loss(dir_logits, labels, pos_weights=self.csp_weights)

        if reg_loss is not None:
            losses['act_reg_loss'] = reg_loss
        return losses

    def _forward(self, vis_output: PrecomputedMinibatch, step=None, epoch=None, **kwargs):
        dir_act_logits, vrepr = self.base_model._forward(vis_output, return_repr=True)
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

        return dir_act_logits, gcn_act_logits, action_labels, reg_loss


class PeyreModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'peyre'

    def __init__(self, dataset: HicoDetSingleHOIsSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.word_embs = WordEmbeddings(source='word2vec', normalize=True)
        obj_word_embs = self.word_embs.get_embeddings(dataset.objects)
        act_word_embs = self.word_embs.get_embeddings(dataset.actions)

        interactions = dataset.full_dataset.interactions  # each is [p, o]
        person_word_emb = np.tile(obj_word_embs[dataset.human_class], reps=[interactions.shape[0], 1])
        hoi_embs = np.concatenate([person_word_emb,
                                   act_word_embs[interactions[:, 0]],
                                   obj_word_embs[interactions[:, 1]]], axis=1)

        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_word_embs), requires_grad=False)
        self.act_word_embs = nn.Parameter(torch.from_numpy(act_word_embs), requires_grad=False)
        self.visual_phrases_embs = nn.Parameter(torch.from_numpy(hoi_embs), requires_grad=False)

        appearance_dim = 300
        self.vis_to_app_mlps = nn.ModuleDict({k: nn.Linear(self.vis_feat_dim, appearance_dim) for k in ['sub', 'obj']})

        spatial_dim = 400
        self.spatial_mlp = nn.Sequential(nn.Linear(8, spatial_dim),
                                         nn.Linear(spatial_dim, spatial_dim))

        # Appearance features to latent representation
        output_dim = 1024
        self.x_to_v_mlps = nn.ModuleDict()
        for k in ['sub', 'obj', 'act', 'vp']:
            input_dim = self.vis_feat_dim if k in ['sub', 'obj'] else (appearance_dim * 2 + spatial_dim)
            self.x_to_v_mlps[k] = nn.Sequential(nn.Linear(input_dim, output_dim),
                                                nn.ReLU(),
                                                nn.Dropout(p=0.5),
                                                nn.Linear(output_dim, output_dim),
                                                )

        # Language features to latent representation
        self.q_to_w_mlps = nn.ModuleDict()
        for k in ['sub', 'obj', 'act', 'vp']:
            input_dim = (3 * self.word_embs.dim) if k == 'vp' else self.word_embs.dim
            self.q_to_w_mlps[k] = nn.Sequential(nn.Linear(input_dim, output_dim),
                                                nn.ReLU(),
                                                nn.Linear(output_dim, output_dim))

        # Analogy transformation
        non_rare_triplets = self.dataset.full_dataset.get_triplets(split=self.dataset.split, which='non rare')
        zero_embs = np.zeros((non_rare_triplets.shape[0], self.word_embs.dim))
        non_rare_triplet_embs = {'sub': person_word_emb, 'act': act_word_embs[non_rare_triplets[:, 1]], 'obj': obj_word_embs[non_rare_triplets[:, 2]]}
        self.non_rare_triplet_embs = nn.ModuleDict({k: nn.Parameter(torch.from_numpy(v), requires_grad=False)
                                                    for k, v in non_rare_triplet_embs.items()})
        non_rare_triplet_visual_phrase_embs = {'sub': np.concatenate([person_word_emb, zero_embs, zero_embs], axis=1),
                                               'act': np.concatenate([zero_embs, act_word_embs[non_rare_triplets[:, 1]], zero_embs], axis=1),
                                               'obj': np.concatenate([zero_embs, zero_embs, obj_word_embs[non_rare_triplets[:, 2]]], axis=1),
                                               }
        self.non_rare_triplet_visual_phrase_embs = nn.ModuleDict({k: nn.Parameter(torch.from_numpy(v), requires_grad=False)
                                                                  for k, v in non_rare_triplet_visual_phrase_embs.items()})

        self.k = 5
        self.alphas = {'sub': 0.1, 'act': 0.8, 'obj': 0.1}

    def _get_losses(self, vis_output: PrecomputedMinibatch, outputs):
        hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits = outputs
        box_labels = vis_output.box_labels

        hoi_subj_labels = box_labels[vis_output.ho_infos_np[:, 1]]
        subj_labels_1hot = hoi_subj_labels.new_zeros((hoi_subj_labels.shape[0], self.dataset.num_objects)).float()
        fg_objs = np.flatnonzero(hoi_subj_labels >= 0)
        subj_labels_1hot[fg_objs, hoi_subj_labels[fg_objs]] = 1

        hoi_obj_labels = box_labels[vis_output.ho_infos_np[:, 2]]
        obj_labels_1hot = hoi_obj_labels.new_zeros((hoi_obj_labels.shape[0], self.dataset.num_objects)).float()
        fg_objs = np.flatnonzero(hoi_obj_labels >= 0)
        obj_labels_1hot[fg_objs, hoi_obj_labels[fg_objs]] = 1

        action_labels = vis_output.action_labels
        hoi_labels = vis_output.hoi_labels

        hoi_subj_loss = bce_loss(hoi_subj_logits, subj_labels_1hot)
        hoi_obj_loss = bce_loss(hoi_obj_logits, obj_labels_1hot)
        act_loss = bce_loss(hoi_act_logits, action_labels)
        hoi_loss = bce_loss(hoi_logits, hoi_labels)
        return {'hoi_subj_loss': hoi_subj_loss, 'hoi_obj_loss': hoi_obj_loss, 'act_loss': act_loss, 'hoi_loss': hoi_loss}

    def _finalize_prediction(self, prediction: Prediction, vis_output: PrecomputedMinibatch, outputs):
        hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits = outputs
        interactions = self.dataset.full_dataset.interactions
        hoi_overall_scores = torch.sigmoid(hoi_subj_logits[:, [self.dataset.human_class]]) * \
                             torch.sigmoid(hoi_obj_logits)[:, interactions[:, 1]] * \
                             torch.sigmoid(hoi_act_logits)[:, interactions[:, 0]] * \
                             torch.sigmoid(hoi_logits)
        assert hoi_overall_scores.shape[0] == vis_output.ho_infos_np.shape[0] and \
               hoi_overall_scores.shape[1] == self.dataset.full_dataset.num_interactions

        prediction.hoi_scores = hoi_overall_scores

    def _forward(self, vis_output: PrecomputedMinibatch, step=None, epoch=None, **kwargs):

        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        hoi_infos = vis_output.ho_infos

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

        subj_v = self.x_to_v_mlps['sub'](box_feats)
        subj_w = F.normalize(self.q_to_w_mlps['sub'](self.obj_word_embs))
        subj_logits = subj_v @ subj_w.t()
        hoi_subj_logits = subj_logits[hoi_hum_inds, :]

        obj_v = self.x_to_v_mlps['obj'](box_feats)
        obj_w = F.normalize(self.q_to_w_mlps['obj'](self.obj_word_embs))
        obj_logits = obj_v @ obj_w.t()
        hoi_obj_logits = obj_logits[hoi_obj_inds, :]

        hoi_subj_appearance = self.vis_to_app_mlps['sub'](box_feats)[hoi_hum_inds, :]
        hoi_obj_appearance = self.vis_to_app_mlps['obj'](box_feats)[hoi_obj_inds, :]

        hoi_act_v = self.x_to_v_mlps['act'](torch.cat([hoi_subj_appearance, hoi_obj_appearance, spatial_info], dim=1))
        hoi_act_w = F.normalize(self.q_to_w_mlps['act'](self.act_word_embs))
        hoi_act_logits = hoi_act_v @ hoi_act_w.t()

        hoi_v = self.x_to_v_mlps['vp'](torch.cat([hoi_subj_appearance, hoi_obj_appearance, spatial_info], dim=1))
        hoi_w = F.normalize(self.q_to_w_mlps['vp'](self.visual_phrases_embs))
        hoi_logits = hoi_v @ hoi_w.t()

        # Analogy transformation
        # TODO
        hoi_part_ws = {'sub': subj_w[hoi_hum_inds, :],
                       'act': hoi_act_w,
                       'obj': obj_w[hoi_obj_inds, :]
                       }
        non_rare_triplet_ws = [F.normalize(self.q_to_w_mlps[k](self.non_rare_triplet_embs[k])) for k in ['sub', 'act', 'obj']]
        g = sum([self.alphas[k] * (hoi_part_ws[k] @ non_rare_triplet_ws[k].t()).clamp(min=0) for k in ['sub', 'act', 'obj']])

        _, indices = torch.topk(g, k=self.k, dim=1, largest=True, sorted=True)
        sampled_triplet_inds = indices[torch.arange(indices.shape[0]), torch.randint(low=0, high=self.k, size=(indices.shape[0],)).to(indices)]

        zero_emb = torch.zeros((hoi_w.shape[0], self.word_embs.dim)).to(hoi_w)
        hoi_part_qs = {'sub': self.obj_word_embs[hoi_hum_inds, :],
                       'act': hoi_act_w,
                       'obj': obj_w[hoi_obj_inds, :]
                       }
        hoi_vp_ws = {'sub': F.normalize(self.q_to_w_mlps['vp'](torch.cat([self.obj_word_embs, zero_emb, zero_emb], dim=1))),
                     'act': F.normalize(self.q_to_w_mlps['vp'](torch.cat([zero_emb, self.act_word_embs, zero_emb], dim=1))),
                     'obj': F.normalize(self.q_to_w_mlps['vp'](torch.cat([zero_emb, zero_emb, self.obj_word_embs], dim=1))),
                     }
        non_rare_triplet_vp_ws = [F.normalize(self.q_to_w_mlps['vp'](self.non_rare_triplet_visual_phrase_embs[k])) for k in ['sub', 'act', 'obj']]

        return hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits
