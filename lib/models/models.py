import pickle

from torch.nn import functional as F

from config import cfg
from lib.dataset.hicodet.pc_hicodet_split import PrecomputedMinibatch, Splits
from lib.models.gcns import *
from lib.models.containers import VisualOutput
from lib.models.generic_model import GenericModel, Prediction
from lib.models.misc import bce_loss, LIS
from lib.dataset.utils import get_noun_verb_adj_mat


class BaseModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'base'

    def __init__(self, dataset: HicoDetHoiSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        hidden_dim = 1024
        self.act_repr_dim = cfg.repr_dim

        self.ho_subj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_objects, hidden_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=cfg.dropout),
                                                nn.Linear(hidden_dim, self.final_repr_dim),
                                                ])
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))

        self.ho_obj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_objects, hidden_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(p=cfg.dropout),
                                               nn.Linear(hidden_dim, self.final_repr_dim),
                                               ])
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))

        self.act_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.dropout),
                                            nn.Linear(hidden_dim, self.final_repr_dim),
                                            ])
        nn.init.xavier_normal_(self.act_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.act_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))

        num_classes = dataset.full_dataset.num_interactions if cfg.phoi else dataset.num_actions
        self.output_mlp = nn.Linear(self.final_repr_dim, num_classes, bias=False)
        torch.nn.init.xavier_normal_(self.output_mlp.weight, gain=1.0)

    @property
    def final_repr_dim(self):
        return self.act_repr_dim

    def _forward(self, vis_output: VisualOutput, batch=None, step=None, epoch=None, return_repr=False, return_obj=False):
        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        hoi_infos = vis_output.ho_infos
        union_boxes_feats = vis_output.hoi_union_boxes_feats

        subj_ho_feats = torch.cat([box_feats[hoi_infos[:, 1], :], boxes_ext[hoi_infos[:, 1], 5:]], dim=1)
        obj_ho_feats = torch.cat([box_feats[hoi_infos[:, 2], :], boxes_ext[hoi_infos[:, 2], 5:]], dim=1)

        ho_subj_repr = self.ho_subj_repr_mlp(subj_ho_feats)
        ho_obj_repr = self.ho_obj_repr_mlp(obj_ho_feats)
        act_repr = self.act_repr_mlp(union_boxes_feats)

        hoi_act_repr = ho_subj_repr + ho_obj_repr + act_repr
        if return_repr:
            if return_obj:
                return hoi_act_repr, ho_obj_repr
            return hoi_act_repr

        output_logits = self.output_mlp(hoi_act_repr)

        if cfg.monitor:
            self.values_to_monitor['ho_subj_repr'] = ho_subj_repr
            self.values_to_monitor['ho_obj_repr'] = ho_obj_repr
            self.values_to_monitor['act_repr'] = act_repr
            self.values_to_monitor['output_logits'] = output_logits
        return output_logits


class NonParamBaseModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'npbase'

    def __init__(self, dataset: HicoDetHoiSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        hidden_dim = 1024
        self.act_repr_dim = cfg.repr_dim

        self.ho_subj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_objects, hidden_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=cfg.dropout),
                                                nn.Linear(hidden_dim, self.final_repr_dim),
                                                ])
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))

        self.ho_obj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_objects, hidden_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(p=cfg.dropout),
                                               nn.Linear(hidden_dim, self.final_repr_dim),
                                               ])
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))

        self.act_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.dropout),
                                            nn.Linear(hidden_dim, self.final_repr_dim),
                                            ])
        nn.init.xavier_normal_(self.act_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.act_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))

        predictors = torch.empty((self.final_repr_dim, dataset.num_actions))
        torch.nn.init.xavier_normal_(predictors, gain=1.0)
        self.predictors = nn.Parameter(predictors, requires_grad=True)

    @property
    def final_repr_dim(self):
        return self.act_repr_dim

    def _forward(self, vis_output: VisualOutput, batch=None, step=None, epoch=None, return_repr=False, return_obj=False):
        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        hoi_infos = vis_output.ho_infos
        union_boxes_feats = vis_output.hoi_union_boxes_feats

        subj_ho_feats = torch.cat([box_feats[hoi_infos[:, 1], :], boxes_ext[hoi_infos[:, 1], 5:]], dim=1)
        obj_ho_feats = torch.cat([box_feats[hoi_infos[:, 2], :], boxes_ext[hoi_infos[:, 2], 5:]], dim=1)

        ho_subj_repr = self.ho_subj_repr_mlp(subj_ho_feats)
        ho_obj_repr = self.ho_obj_repr_mlp(obj_ho_feats)
        act_repr = self.act_repr_mlp(union_boxes_feats)

        hoi_act_repr = ho_subj_repr + ho_obj_repr + act_repr
        if return_repr:
            if return_obj:
                return hoi_act_repr, ho_obj_repr
            return hoi_act_repr

        output_logits = hoi_act_repr @ self.predictors
        return output_logits


class ExtKnowledgeGenericModel(GenericModel):
    def __init__(self, dataset: HicoDetHoiSplit, **kwargs):
        super().__init__(dataset, **kwargs)

        self.nv_adj = get_noun_verb_adj_mat(dataset=dataset, isolate_null=True)

        word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        obj_wembs = word_embs.get_embeddings(dataset.full_dataset.objects, retry='avg')
        pred_wembs = word_embs.get_embeddings(dataset.full_dataset.actions, retry='avg')
        if cfg.aggp:
            for j, pe in enumerate(pred_wembs):
                if j == 0:
                    continue
                new_pred_emb = pe
                for i, oe in enumerate(obj_wembs):
                    if self.nv_adj[i, j]:
                        ope = (pe + oe) / 2
                        ope /= np.linalg.norm(ope)
                        new_pred_emb += ope
                new_pred_emb /= np.linalg.norm(new_pred_emb)
                pred_wembs[j, :] = new_pred_emb
        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_wembs), requires_grad=False)
        self.pred_word_embs = nn.Parameter(torch.from_numpy(pred_wembs), requires_grad=False)
        self.pred_emb_sim = nn.Parameter(self.pred_word_embs @ self.pred_word_embs.t(), requires_grad=False)

        self.zs_enabled = (cfg.seenf >= 0)
        self.load_backbone = len(cfg.hoi_backbone) > 0
        if self.zs_enabled:
            print('Zero-shot enabled.')
            seen_pred_inds = pickle.load(open(cfg.active_classes_file, 'rb'))[Splits.TRAIN.value]['pred']
            unseen_pred_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_actions)) - set(seen_pred_inds.tolist())))
            self.seen_pred_inds = nn.Parameter(torch.tensor(seen_pred_inds), requires_grad=False)
            self.unseen_pred_inds = nn.Parameter(torch.tensor(unseen_pred_inds), requires_grad=False)

            if self.load_backbone:
                ckpt = torch.load(cfg.hoi_backbone)
                self.pretrained_base_model = BaseModel(dataset)
                self.pretrained_base_model.load_state_dict(ckpt['state_dict'])
                self.pretrained_predictors = nn.Parameter(self.pretrained_base_model.output_mlp.weight.detach(), requires_grad=False)  # P x D
                assert len(seen_pred_inds) == self.pretrained_predictors.shape[0]
                # self.torch_trained_pred_inds = nn.Parameter(torch.tensor(self.trained_pred_inds), requires_grad=False)

            if cfg.softl:
                self.obj_act_feasibility = nn.Parameter(self.nv_adj, requires_grad=False)

    def get_soft_labels(self, vis_output: VisualOutput):
        # unseen_action_labels = self.obj_act_feasibility[:, self.unseen_pred_inds][vis_output.box_labels[vis_output.ho_infos_np[:, 2]], :] * 0.75
        # unseen_action_labels = vis_output.boxes_ext[vis_output.ho_infos_np[:, 2], 5:] @ self.obj_act_feasibility[:, self.unseen_pred_inds]
        # unseen_action_labels = self.op_mat[vis_output.box_labels[vis_output.ho_infos_np[:, 2]], :]

        action_labels = vis_output.action_labels
        pred_sims = self.pred_emb_sim[self.seen_pred_inds, :][:, self.unseen_pred_inds].clamp(min=0)
        if cfg.lis:
            pred_sims = LIS(pred_sims, w=18, k=7)
        act_sim = action_labels @ pred_sims / action_labels.sum(dim=1, keepdim=True).clamp(min=1)
        obj_labels = vis_output.box_labels[vis_output.ho_infos_np[:, 2]]
        fg_ho_pair = (obj_labels >= 0)
        unseen_action_labels = act_sim.new_zeros(act_sim.shape)
        unseen_action_labels[fg_ho_pair, :] = act_sim[fg_ho_pair, :] * self.obj_act_feasibility[:, self.unseen_pred_inds][obj_labels[fg_ho_pair], :]
        return unseen_action_labels.detach()

    def _refine_output(self, x: PrecomputedMinibatch, inference, vis_output, outputs):
        if inference and self.load_backbone:
            action_output, action_labels, reg_loss, unseen_action_labels = outputs
            pretrained_vrepr = self.pretrained_base_model._forward(vis_output, return_repr=True).detach()
            pretrained_action_output = pretrained_vrepr @ self.pretrained_predictors.t()  # N x Pt
            action_output[:, self.seen_pred_inds] = pretrained_action_output
            outputs = action_output, action_labels, reg_loss, unseen_action_labels
        return outputs

    def _get_losses(self, vis_output: VisualOutput, outputs):
        action_output, action_labels, reg_loss, unseen_action_labels = outputs
        if cfg.softl > 0:
            assert unseen_action_labels is not None
            unseen_action_logits = action_output[:, self.unseen_pred_inds]
            if cfg.nullzs:
                unseen_action_labels *= (1 - action_labels[:, :1])  # cannot be anything else if it is a positive (i.e., from GT) null

            seen_action_logits = action_output[:, self.seen_pred_inds]
            losses = {'action_loss': bce_loss(seen_action_logits, action_labels),
                      'action_loss_unseen': cfg.softl * bce_loss(unseen_action_logits, unseen_action_labels)}
        else:
            losses = {'action_loss': bce_loss(action_output, action_labels)}
        if reg_loss is not None:
            losses['reg_loss'] = reg_loss
        return losses

    def _finalize_prediction(self, prediction: Prediction, vis_output: VisualOutput, outputs):
        action_output = outputs[0]
        assert not cfg.phoi
        assert action_output.shape[1] == self.dataset.full_dataset.num_actions
        prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()


class ZSBaseModel(ExtKnowledgeGenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'zsb'

    def __init__(self, dataset: HicoDetHoiSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.base_model = BaseModel(dataset, **kwargs)
        assert self.zs_enabled and cfg.softl > 0

        num_classes = dataset.full_dataset.num_actions  # ALL predicates
        self.output_mlp = nn.Linear(self.base_model.final_repr_dim, num_classes, bias=False)
        torch.nn.init.xavier_normal_(self.output_mlp.weight, gain=1.0)

    def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):
        vrepr = self.base_model._forward(vis_output, return_repr=True)
        action_logits = self.output_mlp(vrepr)

        action_labels = vis_output.action_labels
        unseen_action_labels = None
        if action_labels is not None:
            unseen_action_labels = self.get_soft_labels(vis_output)

        reg_loss = None
        return action_logits, action_labels, reg_loss, unseen_action_labels


class ZSGCModel(ExtKnowledgeGenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'zsgc'

    def __init__(self, dataset: HicoDetHoiSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.base_model = BaseModel(dataset, **kwargs)
        self.predictor_dim = 1024

        gcemb_dim = 1024
        if cfg.puregc:
            self.gcn = HicoGCN(dataset, input_dim=gcemb_dim, gc_dims=(gcemb_dim, self.predictor_dim))
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
            self.gcn = HicoGCN(dataset, input_dim=gcemb_dim, gc_dims=(gcemb_dim // 2, latent_dim))

        if cfg.greg > 0:
            self.vv_adj = nn.Parameter((self.nv_adj.t() @ self.nv_adj).clamp(max=1).byte(), requires_grad=False)
            assert (self.vv_adj.diag()[1:] == 1).all()

    def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):
        vrepr = self.base_model._forward(vis_output, return_repr=True)
        _, act_class_embs = self.gcn()  # P x E
        # act_predictors = act_class_embs  # P x D
        act_predictors = self.emb_to_predictor(act_class_embs)  # P x D
        action_logits = vrepr @ act_predictors.t()

        action_labels = vis_output.action_labels
        unseen_action_labels = None
        if action_labels is not None and self.zs_enabled:
            if cfg.softl > 0:
                unseen_action_labels = self.get_soft_labels(vis_output)
            else:  # restrict training to seen predicates only
                action_logits = action_logits[:, self.seen_pred_inds]  # P x E

        reg_loss = None
        if cfg.greg > 0:
            act_predictors_norm = F.normalize(act_predictors, dim=1)
            act_predictors_sim = act_predictors_norm @ act_predictors_norm.t()
            arange = torch.arange(act_predictors_sim.shape[0])

            # Done with argmin/argmax because using min/max directly resulted in NaNs.
            neigh_mask = torch.full_like(act_predictors_sim, np.inf)
            neigh_mask[self.vv_adj] = 1
            argmin_neigh_sim = (act_predictors_sim * neigh_mask.detach()).argmin(dim=1)
            min_neigh_sim = act_predictors_sim[arange, argmin_neigh_sim]

            non_neigh_mask = torch.full_like(act_predictors_sim, -np.inf)
            non_neigh_mask[~self.vv_adj] = 1
            argmax_non_neigh_sim = (act_predictors_sim * non_neigh_mask.detach()).argmax(dim=1)
            max_non_neigh_sim = act_predictors_sim[arange, argmax_non_neigh_sim]

            # Exclude null interaction
            min_neigh_sim = min_neigh_sim[1:]
            max_non_neigh_sim = max_non_neigh_sim[1:]

            assert not torch.isinf(min_neigh_sim).any() and not torch.isinf(max_non_neigh_sim).any()
            assert not torch.isnan(min_neigh_sim).any() and not torch.isnan(max_non_neigh_sim).any()

            reg_loss = F.relu(cfg.greg_margin - min_neigh_sim + max_non_neigh_sim)
            reg_loss = cfg.greg * reg_loss.mean()

        return action_logits, action_labels, reg_loss, unseen_action_labels


class PeyreModel(GenericModel):
    # FIXME this is not 0-shot

    @classmethod
    def get_cline_name(cls):
        return 'peyre'

    def __init__(self, dataset: HicoDetHoiSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.word_embs = WordEmbeddings(source='word2vec', normalize=True)
        obj_word_embs = self.word_embs.get_embeddings(dataset.objects)
        pred_word_embs = self.word_embs.get_embeddings(dataset.predicates)

        interactions = dataset.full_dataset.interactions  # each is [p, o]
        person_word_emb = np.tile(obj_word_embs[dataset.human_class], reps=[interactions.shape[0], 1])
        hoi_embs = np.concatenate([person_word_emb,
                                   pred_word_embs[interactions[:, 0]],
                                   obj_word_embs[interactions[:, 1]]], axis=1)

        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_word_embs), requires_grad=False)
        self.pred_word_embs = nn.Parameter(torch.from_numpy(pred_word_embs), requires_grad=False)
        self.visual_phrases_embs = nn.Parameter(torch.from_numpy(hoi_embs), requires_grad=False)

        appearance_dim = 300
        self.vis_to_app_mlps = nn.ModuleDict({k: nn.Linear(self.visual_module.vis_feat_dim, appearance_dim) for k in ['sub', 'obj']})

        spatial_dim = 400
        self.spatial_mlp = nn.Sequential(nn.Linear(8, spatial_dim),
                                         nn.Linear(spatial_dim, spatial_dim))

        output_dim = 1024
        self.app_to_repr_mlps = nn.ModuleDict()
        for k in ['sub', 'obj', 'pred', 'vp']:
            input_dim = self.visual_module.vis_feat_dim if k in ['sub', 'obj'] else (appearance_dim * 2 + spatial_dim)
            self.app_to_repr_mlps[k] = nn.Sequential(nn.Linear(input_dim, output_dim),
                                                     nn.ReLU(),
                                                     nn.Dropout(p=0.5),
                                                     nn.Linear(output_dim, output_dim),
                                                     )

        self.wemb_to_repr_mlps = nn.ModuleDict()
        for k in ['sub', 'obj', 'pred', 'vp']:
            input_dim = (3 * self.word_embs.dim) if k == 'vp' else self.word_embs.dim
            self.wemb_to_repr_mlps[k] = nn.Sequential(nn.Linear(input_dim, output_dim),
                                                      nn.ReLU(),
                                                      nn.Linear(output_dim, output_dim))

    def _get_losses(self, vis_output: VisualOutput, outputs):
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
        return {'hoi_subj_loss': hoi_subj_loss, 'hoi_obj_loss': hoi_obj_loss, 'action_loss': act_loss, 'hoi_loss': hoi_loss}

    def _finalize_prediction(self, prediction: Prediction, vis_output: VisualOutput, outputs):
        hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits = outputs
        interactions = self.dataset.full_dataset.interactions
        hoi_overall_scores = torch.sigmoid(hoi_subj_logits[:, [self.dataset.human_class]]) * \
                             torch.sigmoid(hoi_obj_logits)[:, interactions[:, 1]] * \
                             torch.sigmoid(hoi_act_logits)[:, interactions[:, 0]] * \
                             torch.sigmoid(hoi_logits)
        assert hoi_overall_scores.shape[0] == vis_output.ho_infos_np.shape[0] and \
               hoi_overall_scores.shape[1] == self.dataset.full_dataset.num_interactions

        prediction.hoi_scores = hoi_overall_scores

    def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):

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

        subj_repr = self.app_to_repr_mlps['sub'](box_feats)
        subj_emb = F.normalize(self.wemb_to_repr_mlps['sub'](self.obj_word_embs))
        subj_logits = subj_repr @ subj_emb.t()
        hoi_subj_logits = subj_logits[hoi_hum_inds, :]

        obj_repr = self.app_to_repr_mlps['obj'](box_feats)
        obj_emb = F.normalize(self.wemb_to_repr_mlps['obj'](self.obj_word_embs))
        obj_logits = obj_repr @ obj_emb.t()
        hoi_obj_logits = obj_logits[hoi_obj_inds, :]

        hoi_subj_appearance = self.vis_to_app_mlps['sub'](box_feats)[hoi_hum_inds, :]
        hoi_obj_appearance = self.vis_to_app_mlps['obj'](box_feats)[hoi_obj_inds, :]

        hoi_act_repr = self.app_to_repr_mlps['pred'](torch.cat([hoi_subj_appearance, hoi_obj_appearance, spatial_info], dim=1))
        hoi_act_emb = F.normalize(self.wemb_to_repr_mlps['pred'](self.pred_word_embs))
        hoi_act_logits = hoi_act_repr @ hoi_act_emb.t()

        hoi_repr = self.app_to_repr_mlps['vp'](torch.cat([hoi_subj_appearance, hoi_obj_appearance, spatial_info], dim=1))
        hoi_emb = F.normalize(self.wemb_to_repr_mlps['vp'](self.visual_phrases_embs))
        hoi_logits = hoi_repr @ hoi_emb.t()

        return hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits
