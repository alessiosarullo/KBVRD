import os
import pickle

from lib.dataset.hicodet.pc_hicodet_split import PrecomputedMinibatch, Splits
from lib.models.containers import VisualOutput
from lib.models.generic_model import GenericModel, Prediction
from lib.models.hoi_branches import *


class BaseModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'base'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        self.act_repr_dim = 1024
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim

        self.ho_subj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.final_repr_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(self.final_repr_dim, self.final_repr_dim),
                                                # nn.ReLU(inplace=True),
                                                # nn.Dropout(0.5),
                                                ])
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.ho_obj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.final_repr_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(0.5),
                                               nn.Linear(self.final_repr_dim, self.final_repr_dim),
                                               # nn.ReLU(inplace=True),
                                               # nn.Dropout(0.5),
                                               ])
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.act_output_fc = nn.Linear(self.final_repr_dim, dataset.num_predicates, bias=False)
        torch.nn.init.xavier_normal_(self.act_output_fc.weight, gain=1.0)

    @property
    def final_repr_dim(self):
        return self.act_repr_dim

    def _forward(self, vis_output: VisualOutput, step=None, epoch=None, return_repr=False, return_obj=False):
        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        masks = vis_output.masks
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

        box_feats_ext = torch.cat([box_feats, boxes_ext[:, 5:]], dim=1)

        ho_subj_repr = self.ho_subj_repr_mlp(box_feats_ext[hoi_infos[:, 1], :])
        ho_obj_repr = self.ho_obj_repr_mlp(box_feats_ext[hoi_infos[:, 2], :])
        act_repr = ho_subj_repr + ho_obj_repr
        if return_repr:
            if return_obj:
                return act_repr, ho_obj_repr
            return act_repr

        action_logits = self.act_output_fc(act_repr)
        return action_logits


class MultiModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'multi'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        self.act_repr_dim = 1024
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim

        # Action through object
        self.act_subj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.final_repr_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(0.5),
                                                 nn.Linear(self.final_repr_dim, self.final_repr_dim),
                                                 ])
        nn.init.xavier_normal_(self.act_subj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.act_subj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.act_subj_output_fc = nn.Linear(self.final_repr_dim, dataset.num_predicates, bias=False)
        torch.nn.init.xavier_normal_(self.act_subj_output_fc.weight, gain=1.0)

        # Action
        self.act_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, self.final_repr_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.5),
                                            nn.Linear(self.final_repr_dim, self.final_repr_dim),
                                            ])
        nn.init.xavier_normal_(self.act_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.act_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.act_output_fc = nn.Linear(self.final_repr_dim, dataset.num_predicates, bias=False)
        torch.nn.init.xavier_normal_(self.act_output_fc.weight, gain=1.0)

        # Action through HOI
        self.hoi_subj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.final_repr_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(0.5),
                                                 nn.Linear(self.final_repr_dim, self.final_repr_dim),
                                                 ])
        nn.init.xavier_normal_(self.hoi_subj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.hoi_subj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.hoi_obj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.final_repr_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(self.final_repr_dim, self.final_repr_dim),
                                                ])
        nn.init.xavier_normal_(self.hoi_obj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.hoi_obj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.hoi_act_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, self.final_repr_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(self.final_repr_dim, self.final_repr_dim),
                                                ])
        nn.init.xavier_normal_(self.hoi_act_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.hoi_act_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.act_hoi_output_fc = nn.Linear(self.final_repr_dim, dataset.num_predicates, bias=False)
        torch.nn.init.xavier_normal_(self.act_hoi_output_fc.weight, gain=1.0)

    @property
    def final_repr_dim(self):
        return self.act_repr_dim

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos is not None:
                act_hoi_output, act_output, act_sub_output = self._forward(vis_output, epoch=x.epoch, step=x.iter)
            else:
                assert inference
                act_hoi_output = act_output = act_sub_output = None

            if not inference:
                action_labels = vis_output.action_labels
                losses = {'action_loss': nn.functional.binary_cross_entropy_with_logits(act_output, action_labels) * act_output.shape[1],
                          'action_subj_loss': nn.functional.binary_cross_entropy_with_logits(act_sub_output, action_labels) * act_sub_output.shape[1],
                          'action_hoi_loss': nn.functional.binary_cross_entropy_with_logits(act_hoi_output, action_labels) * act_hoi_output.shape[1],
                          }
                return losses
            else:
                prediction = Prediction()

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2].cpu().numpy()

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes
                    prediction.obj_scores = boxes_ext[:, 5:]

                    if vis_output.ho_infos is not None:
                        act_score = (torch.sigmoid(act_hoi_output) +
                                     torch.sigmoid(act_output) +
                                     torch.sigmoid(act_sub_output)).cpu().numpy() / 3

                        if act_output.shape[1] < self.dataset.hicodet.num_predicates:
                            raise NotImplementedError

                        prediction.ho_img_inds = vis_output.ho_infos[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos[:, 1:]
                        prediction.action_scores = act_score

                return prediction

    def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):
        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        masks = vis_output.masks
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

        box_feats_ext = torch.cat([box_feats, boxes_ext[:, 5:]], dim=1)
        subj_ho_feats = box_feats_ext[hoi_infos[:, 1], :]
        obj_ho_feats = box_feats_ext[hoi_infos[:, 2], :]

        act_subj_repr = self.act_subj_repr_mlp(subj_ho_feats)
        act_subj_logits = self.act_subj_output_fc(act_subj_repr)

        act_repr = self.act_repr_mlp(subj_ho_feats + obj_ho_feats)
        act_logits = self.act_output_fc(act_repr)

        hoi_subj_repr = self.hoi_subj_repr_mlp(subj_ho_feats)
        hoi_obj_repr = self.hoi_obj_repr_mlp(obj_ho_feats)
        act_hoi_logits = self.act_hoi_output_fc(hoi_subj_repr + hoi_obj_repr)
        return act_hoi_logits, act_logits, act_subj_logits


class ZSBaseModel(GenericModel):
    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.dataset = dataset
        self.base_model = BaseModel(dataset, **kwargs)
        self.predictor_dim = self.base_model.final_repr_dim

        seen_pred_inds = pickle.load(open(cfg.program.active_classes_file, 'rb'))[Splits.TRAIN.value]['pred']
        unseen_pred_inds = np.array(sorted(set(range(self.dataset.hicodet.num_predicates)) - set(seen_pred_inds.tolist())))
        self.seen_pred_inds = nn.Parameter(torch.tensor(seen_pred_inds), requires_grad=False)
        self.unseen_pred_inds = nn.Parameter(torch.tensor(unseen_pred_inds), requires_grad=False)

        if cfg.model.zsload:
            ckpt = torch.load(cfg.program.baseline_model_file)
            self.pretrained_base_model = BaseModel(dataset)
            self.pretrained_base_model.load_state_dict(ckpt['state_dict'])
            self.pretrained_predictors = nn.Parameter(self.pretrained_base_model.act_output_fc.weight.detach(), requires_grad=False)  # P x D
            assert len(seen_pred_inds) == self.pretrained_predictors.shape[0]
            # self.torch_trained_pred_inds = nn.Parameter(torch.tensor(self.trained_pred_inds), requires_grad=False)

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos is not None:
                action_output, action_labels, reg_loss = self._forward(vis_output, epoch=x.epoch, step=x.iter)
                if inference and cfg.model.zsload:
                    pretrained_vrepr = self.pretrained_base_model._forward(vis_output, return_repr=True).detach()
                    pretrained_action_output = pretrained_vrepr @ self.pretrained_predictors.t()  # N x Pt

                    action_output[:, self.seen_pred_inds] = pretrained_action_output
            else:
                assert inference
                action_output = action_labels = None

            if not inference:
                losses = {'action_loss': nn.functional.binary_cross_entropy_with_logits(action_output, action_labels) * action_output.shape[1]}
                if reg_loss is not None:
                    losses['reg_loss'] = reg_loss
                return losses
            else:
                prediction = Prediction()

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2].cpu().numpy()

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes
                    prediction.obj_scores = boxes_ext[:, 5:]

                    if vis_output.ho_infos is not None:
                        assert action_output is not None

                        prediction.ho_img_inds = vis_output.ho_infos[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos[:, 1:]

                        prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()

                return prediction


class ZSHoiModel(ZSBaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'zsh'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        self.emb_dim = 200
        super().__init__(dataset, **kwargs)

        latent_dim = self.emb_dim
        input_dim = self.predictor_dim
        self.vrepr_to_emb = nn.Sequential(*[nn.Linear(input_dim, 800),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.5),
                                            nn.Linear(800, 600),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.5),
                                            nn.Linear(600, 2 * latent_dim),
                                            ])
        self.emb_to_predictor = nn.Sequential(*[nn.Linear(latent_dim, 600),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(600, 800),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(800, input_dim),
                                                ])

        self.gcn = CheatHoiGCNBranch(dataset, input_repr_dim=512, gc_dims=(300, self.emb_dim))
        adj_av = (self.gcn.adj_av > 0).float()
        self.adj_av_norm = nn.Parameter(adj_av / adj_av.sum(dim=1, keepdim=True))

        if cfg.model.softlabels:
            self.obj_act_feasibility = nn.Parameter(self.gcn.noun_verb_links, requires_grad=False)

    def get_soft_labels(self, vis_output: VisualOutput):
        raise NotImplementedError

    def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):
        vrepr = self.base_model._forward(vis_output, return_repr=True)

        hoi_class_embs, _, _ = self.gcn()  # I x E
        if vis_output.action_labels is not None:
            if cfg.model.softlabels:
                action_labels = self.get_soft_labels(vis_output)
            else:  # restrict training to seen predicates only
                action_labels = vis_output.action_labels
        else:
            action_labels = None

        if cfg.model.attw:
            instance_params = self.vrepr_to_emb(vrepr)
            instance_means = instance_params[:, :instance_params.shape[1] // 2]  # P x E
            instance_logstd = instance_params[:, instance_params.shape[1] // 2:]  # P x E
            instance_logstd = instance_logstd.unsqueeze(dim=1)
            instance_class_logprobs = - 0.5 * (2 * instance_logstd.sum(dim=2) +  # NOTE: constant term is missing
                                               ((instance_means.unsqueeze(dim=1) - hoi_class_embs.unsqueeze(dim=0)) / instance_logstd.exp()).norm(
                                                   dim=2) ** 2)
            hoi_predictors = self.emb_to_predictor(instance_class_logprobs.exp().unsqueeze(dim=2) *
                                                   nn.functional.normalize(hoi_class_embs, dim=1).unsqueeze(dim=0))  # N x P x D
            hoi_logits = torch.bmm(vrepr.unsqueeze(dim=1), hoi_predictors.transpose(1, 2)).squeeze(dim=1)
        else:
            hoi_predictors = self.emb_to_predictor(hoi_class_embs)  # P x D
            hoi_logits = vrepr @ hoi_predictors.t()

        ho_obj_inter_prior = vis_output.boxes_ext[vis_output.ho_infos[:, 2], 5:][:, self.dataset.hicodet.interactions[:, 1]]
        hoi_logits = hoi_logits * ho_obj_inter_prior

        act_logits = hoi_logits @ self.adj_av_norm  # get action class embeddings through marginalisation

        if vis_output.action_labels is not None and not cfg.model.softlabels:  # restrict training to seen predicates only
            act_logits = act_logits[:, self.seen_pred_inds]

        reg_loss = None
        return act_logits, action_labels, reg_loss


class ZSModel(ZSBaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'zsgc'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        self.emb_dim = 200
        super().__init__(dataset, **kwargs)

        latent_dim = self.emb_dim
        input_dim = self.predictor_dim
        self.vrepr_to_emb = nn.Sequential(*[nn.Linear(input_dim, 800),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.5),
                                            nn.Linear(800, 600),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.5),
                                            nn.Linear(600, 2 * latent_dim),
                                            ])
        self.emb_to_predictor = nn.Sequential(*[nn.Linear(latent_dim, 600),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(600, 800),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(800, input_dim),
                                                ])

        if cfg.model.oscore:
            self.obj_scores_to_act_logits = nn.Sequential(*[nn.Linear(self.dataset.num_object_classes, self.dataset.hicodet.num_predicates)])

        self.gcn = CheatGCNBranch(dataset, input_repr_dim=512, gc_dims=(300, self.emb_dim))

        if cfg.model.softlabels:
            self.obj_act_feasibility = nn.Parameter(self.gcn.noun_verb_links, requires_grad=False)

    def get_soft_labels(self, vis_output: VisualOutput):
        ho_infos = torch.tensor(vis_output.ho_infos, device=vis_output.action_labels.device)
        action_labels = vis_output.boxes_ext[ho_infos[:, 2], 5:] @ self.obj_act_feasibility

        action_labels[:, self.seen_pred_inds] = vis_output.action_labels

        action_labels = action_labels.detach()
        return action_labels

    def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):
        vrepr = self.base_model._forward(vis_output, return_repr=True)

        _, class_embs = self.gcn()  # P x E
        if vis_output.action_labels is not None:
            if cfg.model.softlabels:
                action_labels = self.get_soft_labels(vis_output)
                act_prior = (self.gcn.noun_verb_links[vis_output.box_labels, :]).clamp(min=1e-8)
            else:  # restrict training to seen predicates only
                class_embs = class_embs[self.seen_pred_inds, :]  # P x E
                action_labels = vis_output.action_labels
                act_prior = (self.gcn.noun_verb_links[vis_output.box_labels, :][:, self.seen_pred_inds]).clamp(min=1e-8)
            act_logprior = act_prior[vis_output.ho_infos[:, 2], :].log()
        else:
            action_labels = None
            obj_max, obj_argmax = vis_output.boxes_ext[vis_output.ho_infos[:, 2], 5:].max(dim=1)
            act_prior = (self.gcn.noun_verb_links[obj_argmax, :] * obj_max.unsqueeze(dim=1)).clamp(min=1e-8)
            act_logprior = act_prior.log()

        if cfg.model.attw:
            instance_params = self.vrepr_to_emb(vrepr)
            instance_means = instance_params[:, :instance_params.shape[1] // 2]  # P x E
            instance_logstd = instance_params[:, instance_params.shape[1] // 2:]  # P x E
            instance_logstd = instance_logstd.unsqueeze(dim=1)
            class_logprobs = - 0.5 * (2 * instance_logstd.sum(dim=2) +  # NOTE: constant term is missing
                                      ((instance_means.unsqueeze(dim=1) - class_embs.unsqueeze(dim=0)) / instance_logstd.exp()).norm(dim=2) ** 2)
            act_predictors = self.emb_to_predictor(class_logprobs.exp().unsqueeze(dim=2) *
                                                   nn.functional.normalize(class_embs, dim=1).unsqueeze(dim=0))  # N x P x D
            action_logits = torch.bmm(vrepr.unsqueeze(dim=1), act_predictors.transpose(1, 2)).squeeze(dim=1)
        else:
            act_predictors = self.emb_to_predictor(class_embs)  # P x D
            action_logits = vrepr @ act_predictors.t()

        if cfg.model.oprior:
            action_logits = action_logits + act_logprior

        if cfg.model.oscore:
            action_logits_from_obj_score = self.obj_scores_to_act_logits(vis_output.boxes_ext[:, 5:])[vis_output.ho_infos[:, 2], :]
            if vis_output.action_labels is not None and not cfg.model.softlabels:
                action_logits_from_obj_score = action_logits_from_obj_score[:, self.seen_pred_inds]  # P x E
            action_logits = action_logits + action_logits_from_obj_score

        reg_loss = None
        return action_logits, action_labels, reg_loss


class ZSObjModel(ZSBaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'zso'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        self.emb_dim = 200
        super().__init__(dataset, **kwargs)

        latent_dim = self.emb_dim
        input_dim = self.predictor_dim
        self.vrepr_to_emb = nn.Sequential(*[nn.Linear(input_dim, 800),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.5),
                                            nn.Linear(800, 600),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.5),
                                            nn.Linear(600, 2 * latent_dim),
                                            ])
        self.emb_to_predictor = nn.Sequential(*[nn.Linear(latent_dim, 600),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(600, 800),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(800, input_dim),
                                                ])
        self.emb_to_obj_predictor = nn.Sequential(*[nn.Linear(latent_dim, 600),
                                                    nn.ReLU(inplace=True),
                                                    nn.Dropout(0.5),
                                                    nn.Linear(600, 800),
                                                    nn.ReLU(inplace=True),
                                                    nn.Dropout(0.5),
                                                    nn.Linear(800, input_dim),
                                                    ])

        self.gcn = CheatGCNBranch(dataset, input_repr_dim=512, gc_dims=(300, self.emb_dim))

        if cfg.model.softlabels:
            self.obj_act_feasibility = nn.Parameter(self.gcn.noun_verb_links, requires_grad=False)

    def get_soft_labels(self, vis_output: VisualOutput):
        ho_infos = torch.tensor(vis_output.ho_infos, device=vis_output.action_labels.device)
        action_labels = vis_output.boxes_ext[ho_infos[:, 2], 5:] @ self.obj_act_feasibility

        action_labels[:, self.seen_pred_inds] = vis_output.action_labels

        action_labels = action_labels.detach()
        return action_labels

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos is not None:
                action_output, action_labels, reg_loss, ho_obj_output, ho_obj_labels = self._forward(vis_output, epoch=x.epoch, step=x.iter)
                if inference and cfg.model.zsload:
                    pretrained_vrepr = self.pretrained_base_model._forward(vis_output, return_repr=True).detach()
                    pretrained_action_output = pretrained_vrepr @ self.pretrained_predictors.t()  # N x Pt

                    action_output[:, self.seen_pred_inds] = pretrained_action_output
            else:
                assert inference
                action_output = action_labels = None

            if not inference:
                losses = {'action_loss': nn.functional.binary_cross_entropy_with_logits(action_output, action_labels) * action_output.shape[1],
                          'obj_loss': nn.functional.cross_entropy(ho_obj_output, ho_obj_labels),
                          }
                if reg_loss is not None:
                    losses['reg_loss'] = reg_loss
                return losses
            else:
                prediction = Prediction()

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2].cpu().numpy()

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes
                    prediction.obj_scores = boxes_ext[:, 5:]

                    if vis_output.ho_infos is not None:
                        assert action_output is not None

                        prediction.ho_img_inds = vis_output.ho_infos[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos[:, 1:]

                        prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()

                return prediction

    def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):
        act_vrepr, ho_obj_vrepr = self.base_model._forward(vis_output, return_repr=True, return_obj=True)
        act_instance_params = self.vrepr_to_emb(act_vrepr)
        act_instance_means = act_instance_params[:, :act_instance_params.shape[1] // 2]  # P x E
        act_instance_logstd = act_instance_params[:, act_instance_params.shape[1] // 2:]  # P x E

        obj_class_embs, act_class_embs = self.gcn()  # P x E
        if vis_output.action_labels is not None:
            obj_predictors = self.emb_to_obj_predictor(obj_class_embs)  # P x D
            ho_obj_output = ho_obj_vrepr @ obj_predictors.t()
            ho_infos = torch.tensor(vis_output.ho_infos, device=ho_obj_vrepr.device)
            ho_obj_labels = vis_output.box_labels[ho_infos[:, 2]]
            fg_objs = (ho_obj_labels >= 0)
            ho_obj_output = ho_obj_output[fg_objs, :]
            ho_obj_labels = ho_obj_labels[fg_objs]

            if cfg.model.softlabels:
                action_labels = self.get_soft_labels(vis_output)
            else:  # restrict training to seen predicates only
                act_class_embs = act_class_embs[self.seen_pred_inds, :]  # P x E
                action_labels = vis_output.action_labels
        else:
            action_labels = ho_obj_output = ho_obj_labels = None

        act_instance_logstd = act_instance_logstd.unsqueeze(dim=1)
        class_logprobs = - 0.5 * (2 * act_instance_logstd.sum(dim=2) +  # NOTE: constant term is missing
                                  ((act_instance_means.unsqueeze(dim=1) - act_class_embs.unsqueeze(dim=0)) /
                                   act_instance_logstd.exp()).norm(dim=2) ** 2)

        if cfg.model.attw:
            act_predictors = self.emb_to_predictor(class_logprobs.exp().unsqueeze(dim=2) *
                                                   nn.functional.normalize(act_class_embs, dim=1).unsqueeze(dim=0))  # N x P x D
            action_output = torch.bmm(act_vrepr.unsqueeze(dim=1), act_predictors.transpose(1, 2)).squeeze(dim=1)
        else:
            act_predictors = self.emb_to_predictor(act_class_embs)  # P x D
            action_output = act_vrepr @ act_predictors.t()

        reg_loss = None
        return action_output, action_labels, reg_loss, ho_obj_output, ho_obj_labels


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
                                                nn.Dropout(0.5),
                                                nn.Linear(self.final_repr_dim, self.final_repr_dim),
                                                ])
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('leaky_relu'))

        self.ho_obj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.final_repr_dim),
                                               nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                               nn.Dropout(0.5),
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

            if vis_output.ho_infos is not None:
                hoi_output = self._forward(vis_output, )
            else:
                assert inference
                hoi_output = None

            if not inference:
                hoi_labels = vis_output.get_hoi_labels(self.dataset)
                losses = {'hoi_loss': nn.functional.binary_cross_entropy_with_logits(hoi_output, hoi_labels) * hoi_output.shape[1]}
                return losses
            else:
                prediction = Prediction()

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2].cpu().numpy()

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes
                    prediction.obj_scores = boxes_ext[:, 5:]

                    if vis_output.ho_infos is not None:
                        assert hoi_output is not None

                        prediction.ho_img_inds = vis_output.ho_infos[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos[:, 1:]
                        prediction.hoi_scores = torch.sigmoid(hoi_output).cpu().numpy()

                return prediction

    def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):

        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        masks = vis_output.masks
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

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

            if vis_output.ho_infos is not None:
                hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits = self._forward(vis_output)
            else:
                assert inference
                hoi_subj_logits = hoi_obj_logits = hoi_act_logits = hoi_logits = None

            if not inference:
                box_labels = vis_output.box_labels

                hoi_subj_labels = box_labels[vis_output.ho_infos[:, 1]]
                subj_labels_1hot = hoi_subj_labels.new_zeros((hoi_subj_labels.shape[0], self.dataset.num_object_classes)).float()
                subj_labels_1hot[torch.arange(subj_labels_1hot.shape[0]), hoi_subj_labels] = 1

                hoi_obj_labels = box_labels[vis_output.ho_infos[:, 2]]
                obj_labels_1hot = hoi_obj_labels.new_zeros((hoi_obj_labels.shape[0], self.dataset.num_object_classes)).float()
                obj_labels_1hot[torch.arange(obj_labels_1hot.shape[0]), hoi_obj_labels] = 1

                action_labels = vis_output.action_labels

                interactions = self.dataset.hicodet.interactions
                hoi_labels = obj_labels_1hot[:, interactions[:, 1]] * action_labels[:, interactions[:, 0]]
                assert hoi_labels.shape[0] == action_labels.shape[0] and hoi_labels.shape[1] == self.dataset.hicodet.num_interactions

                hoi_subj_loss = nn.functional.binary_cross_entropy_with_logits(hoi_subj_logits, subj_labels_1hot)
                hoi_obj_loss = nn.functional.binary_cross_entropy_with_logits(hoi_obj_logits, obj_labels_1hot)
                act_loss = nn.functional.binary_cross_entropy_with_logits(hoi_act_logits, action_labels)
                hoi_loss = nn.functional.binary_cross_entropy_with_logits(hoi_logits, hoi_labels)
                return {'hoi_subj_loss': hoi_subj_loss, 'hoi_obj_loss': hoi_obj_loss, 'action_loss': act_loss, 'hoi_loss': hoi_loss}
            else:
                prediction = Prediction()

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2].cpu().numpy()

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes
                    prediction.obj_scores = boxes_ext[:, 5:]

                    if vis_output.ho_infos is not None:
                        assert hoi_subj_logits is not None and hoi_obj_logits is not None and hoi_act_logits is not None and hoi_logits is not None
                        interactions = self.dataset.hicodet.interactions
                        hoi_overall_scores = torch.sigmoid(hoi_subj_logits[:, self.dataset.human_class]) * \
                                             torch.sigmoid(hoi_obj_logits)[:, interactions[:, 1]] * \
                                             torch.sigmoid(hoi_act_logits)[:, interactions[:, 0]] * \
                                             torch.sigmoid(hoi_logits)
                        assert hoi_overall_scores.shape[0] == vis_output.ho_infos.shape[0] and \
                               hoi_overall_scores.shape[1] == self.dataset.hicodet.num_interactions

                        prediction.ho_img_inds = vis_output.ho_infos[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos[:, 1:]
                        prediction.hoi_scores = hoi_overall_scores

                return prediction

    def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):

        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        hoi_infos = torch.tensor(vis_output.ho_infos, device=box_feats.device)

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
        subj_emb = torch.nn.functional.normalize(self.wemb_to_repr_mlps['sub'](self.obj_word_embs))
        subj_logits = subj_repr @ subj_emb.t()
        hoi_subj_logits = subj_logits[hoi_hum_inds, :]

        obj_appearance = self.vis_to_app_mlps['obj'](box_feats)
        obj_repr = self.app_to_repr_mlps['obj'](obj_appearance)
        obj_emb = torch.nn.functional.normalize(self.wemb_to_repr_mlps['obj'](self.obj_word_embs))
        obj_logits = obj_repr @ obj_emb.t()
        hoi_obj_logits = obj_logits[hoi_obj_inds, :]

        hoi_subj_appearance = subj_appearance[hoi_hum_inds, :]
        hoi_obj_appearance = obj_appearance[hoi_obj_inds, :]
        hoi_act_repr = self.app_to_repr_mlps['pred'](torch.cat([hoi_subj_appearance, hoi_obj_appearance, spatial_info], dim=1))
        hoi_act_emb = torch.nn.functional.normalize(self.wemb_to_repr_mlps['pred'](self.pred_word_embs))
        hoi_act_logits = hoi_act_repr @ hoi_act_emb.t()

        hoi_repr = self.app_to_repr_mlps['vp'](torch.cat([hoi_subj_appearance, hoi_obj_appearance, spatial_info], dim=1))
        hoi_emb = torch.nn.functional.normalize(self.wemb_to_repr_mlps['vp'](self.visual_phrases_embs))
        hoi_logits = hoi_repr @ hoi_emb.t()

        return hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits
