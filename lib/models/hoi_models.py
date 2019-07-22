import pickle
import os

from lib.dataset.hicodet.pc_hicodet_split import PrecomputedMinibatch, Splits
from lib.models.containers import VisualOutput
from lib.models.generic_model import GenericModel, Prediction, F
from lib.models.hoi_branches import *


def LIS(x, w=None, k=None, T=None):  # defaults are as in the paper
    if T is None:
        if w is None and k is None:
            w, k, T = 10, 12, 8.4
        else:
            assert w is not None and k is not None
            # This is basically what it is: a normalisation constant for when x=1.
            T = 1 + np.exp(k - w).item()
    assert w is not None and k is not None and T is not None
    return T * torch.sigmoid(w * x - k)


class BaseModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'base'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        hidden_dim = 1024
        self.act_repr_dim = cfg.model.repr_dim

        self.ho_subj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, hidden_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=cfg.model.dropout),
                                                nn.Linear(hidden_dim, self.final_repr_dim),
                                                ])
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))

        self.ho_obj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, hidden_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(p=cfg.model.dropout),
                                               nn.Linear(hidden_dim, self.final_repr_dim),
                                               ])
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))

        self.act_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, hidden_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.model.dropout),
                                            nn.Linear(hidden_dim, self.final_repr_dim),
                                            ])
        nn.init.xavier_normal_(self.act_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.act_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('linear'))

        num_classes = dataset.hicodet.num_interactions if cfg.model.phoi else dataset.num_predicates
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

        if cfg.program.monitor:
            self.values_to_monitor['ho_subj_repr'] = ho_subj_repr
            self.values_to_monitor['ho_obj_repr'] = ho_obj_repr
            self.values_to_monitor['act_repr'] = act_repr
            self.values_to_monitor['output_logits'] = output_logits
        return output_logits


class BGFilter(BaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'bg'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.bg_vis_mlp = nn.Sequential(*[nn.Linear(self.visual_module.vis_feat_dim, 1024),
                                          nn.ReLU(inplace=True),
                                          nn.Dropout(p=cfg.model.dropout),
                                          nn.Linear(1024, 512),
                                          ])
        # self.bg_geo_obj_mlp = nn.Sequential(*[nn.Linear(14 + self.dataset.num_object_classes, 128),
        #                                       nn.Linear(128, 256),
        #                                       ])
        #
        # self.bg_detection_mlp = nn.Linear(512 + 256, 1, bias=False)
        self.bg_detection_mlp = nn.Linear(512, 1, bias=False)
        torch.nn.init.xavier_normal_(self.bg_detection_mlp.weight, gain=1.0)

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos_np is not None:
                action_output, bg_output = self._forward(vis_output, batch=x, epoch=x.epoch, step=x.iter)
            else:
                assert inference
                action_output = bg_output = None

            if not inference:
                action_labels = vis_output.action_labels
                if cfg.data.null_as_bg:
                    bg_label = action_labels[:, 0]
                    max_fg_score = torch.sigmoid((action_output * action_labels)[:, 1:].max(dim=1)[0]).detach()
                    act_loss = F.binary_cross_entropy_with_logits(action_output[:, 1:], action_labels[:, 1:]) * (action_output.shape[1] - 1)
                else:
                    bg_label = 1 - (action_labels > 0).any(dim=1).float()
                    max_fg_score = torch.sigmoid((action_output * action_labels).max(dim=1)[0]).detach()
                    act_loss = F.binary_cross_entropy_with_logits(action_output, action_labels) * action_output.shape[1]

                losses = {'action_loss': act_loss}
                if cfg.opt.margin > 0:
                    bg_score = torch.sigmoid(bg_output).squeeze(dim=1)
                    bg_loss = F.margin_ranking_loss(bg_score, max_fg_score, 2 * bg_label - 1, margin=cfg.opt.margin, reduction='none').mean()
                else:
                    bg_loss = F.binary_cross_entropy_with_logits(bg_output, bg_label[:, None])
                losses['bg_loss'] = cfg.opt.bg_coeff * bg_loss
                return losses
            else:
                prediction = Prediction()

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2]

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes
                    prediction.obj_scores = boxes_ext[:, 5:]

                    if vis_output.ho_infos_np is not None:
                        assert action_output is not None

                        if action_output.shape[1] < self.dataset.hicodet.num_predicates:
                            assert action_output.shape[1] == self.dataset.num_predicates
                            restricted_action_output = action_output
                            action_output = restricted_action_output.new_zeros((action_output.shape[0], self.dataset.hicodet.num_predicates))
                            action_output[:, self.dataset.active_predicates] = restricted_action_output

                        action_scores = torch.sigmoid(action_output).cpu().numpy()
                        bg_scores = torch.sigmoid(bg_output).squeeze(dim=1).cpu().numpy()
                        action_scores[:, 0] = bg_scores

                        if cfg.model.filter:
                            # keep = (action_scores[:, 1:].max(axis=1) > bg_scores)
                            keep = (bg_scores < 0.95)

                            if np.any(keep):
                                prediction.ho_img_inds = vis_output.ho_infos_np[keep, 0]
                                prediction.ho_pairs = vis_output.ho_infos_np[keep, 1:]
                                prediction.action_scores = action_scores[keep, :]
                        else:
                            action_scores[:, 1:] *= (1 - bg_scores[:, None])
                            prediction.ho_img_inds = vis_output.ho_infos_np[:, 0]
                            prediction.ho_pairs = vis_output.ho_infos_np[:, 1:]
                            prediction.action_scores = action_scores

                return prediction

    def _forward(self, vis_output: VisualOutput, batch=None, step=None, epoch=None, **kwargs):
        boxes_ext = vis_output.boxes_ext
        ho_infos = vis_output.ho_infos
        union_boxes_feats = vis_output.hoi_union_boxes_feats

        # geo_feats = self.get_geo_feats(vis_output, batch)

        vis_repr = self.bg_vis_mlp(union_boxes_feats)
        # geo_obj_repr = self.bg_geo_obj_mlp(torch.cat([geo_feats, boxes_ext[ho_infos[:, 2], 5:]], dim=1))

        # bg_logits = self.bg_detection_mlp(torch.cat([vis_repr, geo_obj_repr], dim=1))
        bg_logits = self.bg_detection_mlp(vis_repr)

        action_logits = super()._forward(vis_output, batch, step, epoch, **kwargs)
        return action_logits, bg_logits


class ZSGenericModel(GenericModel):
    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.dataset = dataset
        self.base_model = BaseModel(dataset, **kwargs)

        seen_pred_inds = pickle.load(open(cfg.program.active_classes_file, 'rb'))[Splits.TRAIN.value]['pred']
        unseen_pred_inds = np.array(sorted(set(range(self.dataset.hicodet.num_predicates)) - set(seen_pred_inds.tolist())))
        self.seen_pred_inds = nn.Parameter(torch.tensor(seen_pred_inds), requires_grad=False)
        self.unseen_pred_inds = nn.Parameter(torch.tensor(unseen_pred_inds), requires_grad=False)

        self.load_backbone = len(cfg.model.hoi_backbone) > 0
        if self.load_backbone:
            ckpt = torch.load(cfg.model.hoi_backbone)
            self.pretrained_base_model = BaseModel(dataset)
            self.pretrained_base_model.load_state_dict(ckpt['state_dict'])
            self.pretrained_predictors = nn.Parameter(self.pretrained_base_model.output_mlp.weight.detach(), requires_grad=False)  # P x D
            assert len(seen_pred_inds) == self.pretrained_predictors.shape[0]
            # self.torch_trained_pred_inds = nn.Parameter(torch.tensor(self.trained_pred_inds), requires_grad=False)

        word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        obj_wembs = word_embs.get_embeddings(dataset.hicodet.objects, retry='avg')
        pred_wembs = word_embs.get_embeddings(dataset.hicodet.predicates, retry='avg')
        if cfg.model.aggp:
            for j, pe in enumerate(pred_wembs):
                if j == 0:
                    continue
                new_pred_emb = pe
                for i, oe in enumerate(obj_wembs):
                    if self.gcn.noun_verb_links[i, j]:
                        ope = (pe + oe) / 2
                        ope /= np.linalg.norm(ope)
                        new_pred_emb += ope
                new_pred_emb /= np.linalg.norm(new_pred_emb)
                pred_wembs[j, :] = new_pred_emb
        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_wembs), requires_grad=False)
        self.pred_word_embs = nn.Parameter(torch.from_numpy(pred_wembs), requires_grad=False)
        self.pred_emb_sim = nn.Parameter(self.pred_word_embs @ self.pred_word_embs.t(), requires_grad=False)

        if cfg.model.softl:
            self.obj_act_feasibility = nn.Parameter(get_noun_verb_adj_mat(dataset=dataset), requires_grad=False)

    def get_soft_labels(self, vis_output: VisualOutput):
        # unseen_action_labels = self.obj_act_feasibility[:, self.unseen_pred_inds][vis_output.box_labels[vis_output.ho_infos_np[:, 2]], :] * 0.75
        # unseen_action_labels = vis_output.boxes_ext[vis_output.ho_infos_np[:, 2], 5:] @ self.obj_act_feasibility[:, self.unseen_pred_inds]
        # unseen_action_labels = self.op_mat[vis_output.box_labels[vis_output.ho_infos_np[:, 2]], :]

        action_labels = vis_output.action_labels
        pred_sims = self.pred_emb_sim[self.seen_pred_inds, :][:, self.unseen_pred_inds]
        if cfg.model.lis:
            act_sim = action_labels @ LIS(pred_sims.clamp(min=0), w=18, k=7) / action_labels.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            # act_sim = torch.sigmoid(action_labels @ pred_sims)
            act_sim = action_labels @ pred_sims.clamp(min=0) / action_labels.sum(dim=1, keepdim=True).clamp(min=1)
        unseen_action_labels = act_sim * self.obj_act_feasibility[:, self.unseen_pred_inds][vis_output.box_labels[vis_output.ho_infos_np[:, 2]], :]
        return unseen_action_labels.detach()

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos_np is not None:
                action_output, action_labels, reg_loss, unseen_action_labels = self._forward(vis_output, epoch=x.epoch, step=x.iter)
                if inference and self.load_backbone:
                    pretrained_vrepr = self.pretrained_base_model._forward(vis_output, return_repr=True).detach()
                    pretrained_action_output = pretrained_vrepr @ self.pretrained_predictors.t()  # N x Pt

                    action_output[:, self.seen_pred_inds] = pretrained_action_output
            else:
                assert inference
                action_output = action_labels = None

            if not inference:
                if cfg.model.softl > 0:
                    assert unseen_action_labels is not None
                    unseen_action_logits = action_output[:, self.unseen_pred_inds]
                    if cfg.model.nullzs:
                        unseen_action_labels *= (1 - action_labels[:, :1])  # cannot be anything else if it is a positive (i.e., from GT) null

                    seen_action_logits = action_output[:, self.seen_pred_inds]
                    losses = {'action_loss': F.binary_cross_entropy_with_logits(seen_action_logits, action_labels) * seen_action_logits.shape[1],
                              'action_loss_unseen': cfg.model.softl *
                                                    F.binary_cross_entropy_with_logits(unseen_action_logits, unseen_action_labels) *
                                                    unseen_action_labels.shape[1]}
                else:
                    losses = {'action_loss': F.binary_cross_entropy_with_logits(action_output, action_labels) * action_output.shape[1]}
                if reg_loss is not None:
                    losses['reg_loss'] = reg_loss
                return losses
            else:
                prediction = Prediction()

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2]

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes
                    prediction.obj_scores = boxes_ext[:, 5:]

                    if vis_output.ho_infos_np is not None:
                        assert action_output is not None

                        prediction.ho_img_inds = vis_output.ho_infos_np[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos_np[:, 1:]

                        prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()

                return prediction


class ZSBaseModel(ZSGenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'zsb'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        assert cfg.model.softl > 0

        num_classes = dataset.hicodet.num_predicates  # ALL predicates
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


class ZSGCModel(ZSGenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'zsgc'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.predictor_dim = self.base_model.final_repr_dim

        if cfg.model.large:
            gcemb_dim = 2048
            self.emb_dim = 512
        else:
            gcemb_dim = 1024
            self.emb_dim = 200

        latent_dim = self.emb_dim
        input_dim = self.predictor_dim
        self.vrepr_to_emb = nn.Sequential(*[nn.Linear(input_dim, 800),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.model.dropout),
                                            nn.Linear(800, 600),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(p=cfg.model.dropout),
                                            nn.Linear(600, 2 * latent_dim),
                                            ])
        self.emb_to_predictor = nn.Sequential(*[nn.Linear(latent_dim, 600),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=cfg.model.dropout),
                                                nn.Linear(600, 800),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(p=cfg.model.dropout),
                                                nn.Linear(800, input_dim),
                                                ])

        if cfg.model.oscore:
            self.obj_scores_to_act_logits = nn.Sequential(*[nn.Linear(self.dataset.num_object_classes, self.dataset.hicodet.num_predicates)])

        if cfg.model.vv:
            assert not cfg.model.iso_null, 'Not supported'
            self.gcn = ExtCheatGCNBranch(dataset, input_repr_dim=gcemb_dim, gc_dims=(gcemb_dim // 2, self.emb_dim))
        else:
            self.gcn = CheatGCNBranch(dataset, input_repr_dim=gcemb_dim, gc_dims=(gcemb_dim // 2, self.emb_dim))

        if cfg.model.aereg > 0:
            if cfg.model.regsmall:
                self.emb_to_wemb = nn.Linear(latent_dim, self.pred_word_embs.shape[1])
            else:
                self.emb_to_wemb = nn.Sequential(*[nn.Linear(latent_dim, latent_dim),
                                                   nn.ReLU(inplace=True),
                                                   nn.Linear(latent_dim, self.pred_word_embs.shape[1]),
                                                   ])

        op_mat = np.zeros([dataset.hicodet.num_object_classes, dataset.hicodet.num_predicates], dtype=np.float32)
        for _, p, o in dataset.hoi_triplets:
            op_mat[o, p] += 1
        op_mat /= np.maximum(1, op_mat.sum(axis=1, keepdims=True))
        self.op_mat = nn.Parameter(torch.from_numpy(op_mat)[:, self.unseen_pred_inds], requires_grad=False)

        if cfg.model.softl:
            self.obj_act_feasibility = nn.Parameter(self.gcn.noun_verb_links, requires_grad=False)

    def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):
        vrepr = self.base_model._forward(vis_output, return_repr=True)

        _, all_class_embs = self.gcn()  # P x E
        class_embs = all_class_embs
        action_labels = vis_output.action_labels
        unseen_action_labels = None
        if action_labels is not None:
            if cfg.model.softl > 0:
                unseen_action_labels = self.get_soft_labels(vis_output)
            else:  # restrict training to seen predicates only
                class_embs = all_class_embs[self.seen_pred_inds, :]  # P x E

        if cfg.model.attw:
            instance_params = self.vrepr_to_emb(vrepr)
            instance_means = instance_params[:, :instance_params.shape[1] // 2]  # P x E
            instance_logstd = instance_params[:, instance_params.shape[1] // 2:]  # P x E
            instance_logstd = instance_logstd.unsqueeze(dim=1)
            class_logprobs = - 0.5 * (2 * instance_logstd.sum(dim=2) +  # NOTE: constant term is missing
                                      ((instance_means.unsqueeze(dim=1) - class_embs.unsqueeze(dim=0)) / instance_logstd.exp()).norm(dim=2) ** 2)
            act_predictors = self.emb_to_predictor(class_logprobs.exp().unsqueeze(dim=2) *
                                                   F.normalize(class_embs, dim=1).unsqueeze(dim=0))  # N x P x D
            action_logits = torch.bmm(vrepr.unsqueeze(dim=1), act_predictors.transpose(1, 2)).squeeze(dim=1)
        else:
            act_predictors = self.emb_to_predictor(class_embs)  # P x D
            action_logits = vrepr @ act_predictors.t()

        if cfg.model.oprior:
            if vis_output.action_labels is not None:
                if cfg.model.softl:
                    act_prior = (self.gcn.noun_verb_links[vis_output.box_labels, :]).clamp(min=1e-8)
                else:
                    act_prior = (self.gcn.noun_verb_links[vis_output.box_labels, :][:, self.seen_pred_inds]).clamp(min=1e-8)
                act_prior = act_prior[vis_output.ho_infos_np[:, 2], :]
            else:
                obj_max, obj_argmax = vis_output.boxes_ext[vis_output.ho_infos_np[:, 2], 5:].max(dim=1)
                act_prior = (self.gcn.noun_verb_links[obj_argmax, :] * obj_max.unsqueeze(dim=1)).clamp(min=1e-8)
            action_logits = action_logits + act_prior.log()

        if cfg.model.oscore:
            action_logits_from_obj_score = self.obj_scores_to_act_logits(vis_output.boxes_ext[vis_output.ho_infos_np[:, 2], 5:])
            if vis_output.action_labels is not None and not cfg.model.softl:
                action_logits_from_obj_score = action_logits_from_obj_score[:, self.seen_pred_inds]  # P x E
            action_logits = action_logits + action_logits_from_obj_score

        if cfg.model.aereg > 0:
            reg_loss = -cfg.model.aereg * (F.normalize(self.emb_to_wemb(all_class_embs)) * self.pred_word_embs).sum(dim=1).mean()
        else:
            reg_loss = None
        return action_logits, action_labels, reg_loss, unseen_action_labels


class ZSSimModel(ZSBaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'zss'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        assert cfg.model.softl

        seen_pred_inds = pickle.load(open(cfg.program.active_classes_file, 'rb'))[Splits.TRAIN.value]['pred']
        seen_transfer_pred_inds = pickle.load(open(cfg.program.active_classes_file, 'rb'))[Splits.TRAIN.value]['pred_transfer']
        seen_train_pred_inds = np.array([p for p in seen_pred_inds if p not in seen_transfer_pred_inds])
        all_transfer_pred_inds = np.array(sorted(set(range(self.dataset.hicodet.num_predicates)) - set(seen_train_pred_inds.tolist())))
        self.seen_train_inds = nn.Parameter(torch.from_numpy(seen_train_pred_inds), requires_grad=False)
        self.seen_transfer_inds = nn.Parameter(torch.from_numpy(seen_transfer_pred_inds), requires_grad=False)
        self.all_transfer_inds = nn.Parameter(torch.from_numpy(all_transfer_pred_inds), requires_grad=False)

        # these are RELATIVE to the seen ones
        rel_seen_transfer_pred_inds = sorted([np.flatnonzero(seen_pred_inds == p)[0] for p in seen_transfer_pred_inds])
        rel_seen_train_pred_inds = sorted(set(range(len(seen_pred_inds))) - set(rel_seen_transfer_pred_inds))
        self.rel_seen_transfer_inds = nn.Parameter(torch.from_numpy(np.array(rel_seen_transfer_pred_inds)), requires_grad=False)
        self.rel_seen_train_inds = nn.Parameter(torch.from_numpy(np.array(rel_seen_train_pred_inds)), requires_grad=False)

        wemb_dim = self.pred_word_embs.shape[1]
        self.soft_labels_emb_mlp = nn.Sequential(*[nn.Linear(wemb_dim * 2, wemb_dim * 2),
                                                   nn.ReLU(inplace=True),
                                                   # nn.Dropout(p=cfg.model.dropout),
                                                   nn.Linear(wemb_dim * 2, wemb_dim),
                                                   ])

    def get_soft_labels(self, vis_output: VisualOutput):
        # unseen_action_labels = self.obj_act_feasibility[:, self.unseen_pred_inds][vis_output.box_labels[vis_output.ho_infos_np[:, 2]], :] * 0.75
        # unseen_action_labels = vis_output.boxes_ext[vis_output.ho_infos_np[:, 2], 5:] @ self.obj_act_feasibility[:, self.unseen_pred_inds]
        # unseen_action_labels = self.op_mat[vis_output.box_labels[vis_output.ho_infos_np[:, 2]], :]

        known_labels = vis_output.action_labels
        train_labels = known_labels[:, self.rel_seen_train_inds]
        unseen_action_embs = self.soft_labels_emb_mlp(torch.cat([self.obj_word_embs[vis_output.box_labels[vis_output.ho_infos_np[:, 2]], :],
                                                                 train_labels @ self.pred_word_embs[self.seen_train_inds, :]], dim=1))

        # these are for ALL actions
        action_labels_mask = self.obj_act_feasibility[vis_output.box_labels[vis_output.ho_infos_np[:, 2]], :]
        surrogate_action_labels = (F.normalize(unseen_action_embs, dim=1) @ self.pred_word_embs.t()) * action_labels_mask
        surrogate_action_labels = LIS(surrogate_action_labels, w=18, k=7)
        # act_sim = action_labels @ LIS(pred_sims.clamp(min=0), w=18, k=7) / action_labels.sum(dim=1, keepdim=True).clamp(min=1)

        # Loss is for transfer only, actual labels for unseen only
        transfer_labels = known_labels[:, self.rel_seen_transfer_inds].detach()
        unseen_action_label_loss = F.binary_cross_entropy(surrogate_action_labels[:, self.seen_transfer_inds], transfer_labels)

        unseen_action_labels = surrogate_action_labels[:, self.unseen_pred_inds]
        return unseen_action_labels.detach(), unseen_action_label_loss

    def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):
        vrepr = self.base_model._forward(vis_output, return_repr=True)

        action_labels = vis_output.action_labels
        unseen_action_labels, unseen_action_label_loss = None, None
        if action_labels is not None:
            unseen_action_labels, unseen_action_label_loss = self.get_soft_labels(vis_output)

        reg_loss = unseen_action_label_loss  # FIXME hack
        action_logits = self.output_mlp(vrepr)
        return action_logits, action_labels, reg_loss, unseen_action_labels


class KatoModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'kato'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        self._hoi_repr_dim = 600
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim

        self.ho_subj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.final_repr_dim),
                                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                nn.Dropout(p=cfg.model.dropout),
                                                nn.Linear(self.final_repr_dim, self.final_repr_dim),
                                                ])
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.ho_obj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.final_repr_dim),
                                               nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                               nn.Dropout(p=cfg.model.dropout),
                                               nn.Linear(self.final_repr_dim, self.final_repr_dim),
                                               ])
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.gcn_branch = KatoGCNBranch(dataset, self._hoi_repr_dim, gc_dims=(512, 200))

    @property
    def final_repr_dim(self):
        return self._hoi_repr_dim

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos_np is not None:
                hoi_output = self._forward(vis_output, )
            else:
                assert inference
                hoi_output = None

            if not inference:
                hoi_labels = vis_output.hoi_labels
                losses = {'hoi_loss': F.binary_cross_entropy_with_logits(hoi_output, hoi_labels) * hoi_output.shape[1]}
                return losses
            else:
                prediction = Prediction()

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2]

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes
                    prediction.obj_scores = boxes_ext[:, 5:]

                    if vis_output.ho_infos_np is not None:
                        assert hoi_output is not None

                        prediction.ho_img_inds = vis_output.ho_infos_np[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos_np[:, 1:]
                        prediction.hoi_scores = torch.sigmoid(hoi_output).cpu().numpy()

                return prediction

    def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):

        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        hoi_infos = vis_output.ho_infos

        box_feats_ext = torch.cat([box_feats, boxes_ext[:, 5:]], dim=1)

        ho_subj_repr = self.ho_subj_repr_mlp(box_feats_ext[hoi_infos[:, 1], :])
        ho_obj_repr = self.ho_obj_repr_mlp(box_feats_ext[hoi_infos[:, 2], :])
        hoi_repr = ho_subj_repr + ho_obj_repr

        hoi_logits = self.gcn_branch(hoi_repr)
        return hoi_logits


class PeyreModel(GenericModel):
    # FIXME this is not 0-shot

    @classmethod
    def get_cline_name(cls):
        return 'peyre'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.word_embs = WordEmbeddings(source='word2vec', normalize=True)
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
        self.vis_to_app_mlps = nn.ModuleDict({k: nn.Linear(self.visual_module.vis_feat_dim, appearance_dim) for k in ['sub', 'obj']})

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

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos_np is not None:
                hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits = self._forward(vis_output)
            else:
                assert inference
                hoi_subj_logits = hoi_obj_logits = hoi_act_logits = hoi_logits = None

            if not inference:
                box_labels = vis_output.box_labels

                hoi_subj_labels = box_labels[vis_output.ho_infos_np[:, 1]]
                subj_labels_1hot = hoi_subj_labels.new_zeros((hoi_subj_labels.shape[0], self.dataset.num_object_classes)).float()
                subj_labels_1hot[torch.arange(subj_labels_1hot.shape[0]), hoi_subj_labels] = 1

                hoi_obj_labels = box_labels[vis_output.ho_infos_np[:, 2]]
                obj_labels_1hot = hoi_obj_labels.new_zeros((hoi_obj_labels.shape[0], self.dataset.num_object_classes)).float()
                obj_labels_1hot[torch.arange(obj_labels_1hot.shape[0]), hoi_obj_labels] = 1

                action_labels = vis_output.action_labels

                interactions = self.dataset.hicodet.interactions
                hoi_labels = obj_labels_1hot[:, interactions[:, 1]] * action_labels[:, interactions[:, 0]]
                assert hoi_labels.shape[0] == action_labels.shape[0] and hoi_labels.shape[1] == self.dataset.hicodet.num_interactions

                hoi_subj_loss = self.bce_loss(hoi_subj_logits, subj_labels_1hot)
                hoi_obj_loss = self.bce_loss(hoi_obj_logits, obj_labels_1hot)
                act_loss = self.bce_loss(hoi_act_logits, action_labels)
                hoi_loss = self.bce_loss(hoi_logits, hoi_labels)
                return {'hoi_subj_loss': hoi_subj_loss, 'hoi_obj_loss': hoi_obj_loss, 'action_loss': act_loss, 'hoi_loss': hoi_loss}
            else:
                prediction = Prediction()

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2]

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes
                    prediction.obj_scores = boxes_ext[:, 5:]

                    if vis_output.ho_infos_np is not None:
                        assert hoi_subj_logits is not None and hoi_obj_logits is not None and hoi_act_logits is not None and hoi_logits is not None
                        interactions = self.dataset.hicodet.interactions
                        hoi_overall_scores = torch.sigmoid(hoi_subj_logits[:, [self.dataset.human_class]]) * \
                                             torch.sigmoid(hoi_obj_logits)[:, interactions[:, 1]] * \
                                             torch.sigmoid(hoi_act_logits)[:, interactions[:, 0]] * \
                                             torch.sigmoid(hoi_logits)
                        assert hoi_overall_scores.shape[0] == vis_output.ho_infos_np.shape[0] and \
                               hoi_overall_scores.shape[1] == self.dataset.hicodet.num_interactions

                        prediction.ho_img_inds = vis_output.ho_infos_np[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos_np[:, 1:]
                        prediction.hoi_scores = hoi_overall_scores

                return prediction

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

        subj_appearance = self.vis_to_app_mlps['sub'](box_feats)
        subj_repr = self.app_to_repr_mlps['sub'](subj_appearance)
        subj_emb = F.normalize(self.wemb_to_repr_mlps['sub'](self.obj_word_embs))
        subj_logits = subj_repr @ subj_emb.t()
        hoi_subj_logits = subj_logits[hoi_hum_inds, :]

        obj_appearance = self.vis_to_app_mlps['obj'](box_feats)
        obj_repr = self.app_to_repr_mlps['obj'](obj_appearance)
        obj_emb = F.normalize(self.wemb_to_repr_mlps['obj'](self.obj_word_embs))
        obj_logits = obj_repr @ obj_emb.t()
        hoi_obj_logits = obj_logits[hoi_obj_inds, :]

        hoi_subj_appearance = subj_appearance[hoi_hum_inds, :]
        hoi_obj_appearance = obj_appearance[hoi_obj_inds, :]
        hoi_act_repr = self.app_to_repr_mlps['pred'](torch.cat([hoi_subj_appearance, hoi_obj_appearance, spatial_info], dim=1))
        hoi_act_emb = F.normalize(self.wemb_to_repr_mlps['pred'](self.pred_word_embs))
        hoi_act_logits = hoi_act_repr @ hoi_act_emb.t()

        hoi_repr = self.app_to_repr_mlps['vp'](torch.cat([hoi_subj_appearance, hoi_obj_appearance, spatial_info], dim=1))
        hoi_emb = F.normalize(self.wemb_to_repr_mlps['vp'](self.visual_phrases_embs))
        hoi_logits = hoi_repr @ hoi_emb.t()

        return hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits
