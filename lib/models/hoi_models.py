import os

from lib.dataset.hicodet.pc_hicodet_split import PrecomputedMinibatch
from lib.models.containers import VisualOutput
from lib.models.generic_model import GenericModel, Prediction
from lib.models.hoi_branches import *


class BaseModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'base'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self._act_repr_dim = 600

        self.ho_subj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.act_repr_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(self.act_repr_dim, self.act_repr_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                ])
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.ho_obj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.act_repr_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(0.5),
                                               nn.Linear(self.act_repr_dim, self.act_repr_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(0.5),
                                               ])
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.union_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, self.act_repr_dim),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(0.5),
                                              nn.Linear(self.act_repr_dim, self.act_repr_dim),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(0.5),
                                              ])
        nn.init.xavier_normal_(self.union_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.union_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.act_output_fc = nn.Linear(self.act_repr_dim, dataset.num_predicates, bias=False)
        torch.nn.init.xavier_normal_(self.act_output_fc.weight, gain=1.0)

    @property
    def act_repr_dim(self):
        return self._act_repr_dim

    def _forward(self, vis_output: VisualOutput, return_repr=False):
        if vis_output.box_labels is not None:
            vis_output.filter_boxes()
        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        masks = vis_output.masks
        union_boxes_feats = vis_output.hoi_union_boxes_feats
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

        box_feats_ext = torch.cat([box_feats, boxes_ext[:, 5:]], dim=1)

        ho_subj_repr = self.ho_subj_repr_mlp(box_feats_ext[hoi_infos[:, 1], :])
        ho_obj_repr = self.ho_obj_repr_mlp(box_feats_ext[hoi_infos[:, 2], :])
        union_repr = self.union_repr_mlp(union_boxes_feats)
        act_repr = union_repr + ho_subj_repr + ho_obj_repr
        if return_repr:
            return act_repr

        action_logits = self.act_output_fc(act_repr)
        return action_logits


class MultiModalModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'mm'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self._act_repr_dim = 600

        self.ho_subj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.act_repr_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(self.act_repr_dim, self.act_repr_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                ])
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_subj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.ho_obj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.act_repr_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(0.5),
                                               nn.Linear(self.act_repr_dim, self.act_repr_dim),
                                               nn.ReLU(inplace=True),
                                               nn.Dropout(0.5),
                                               ])
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.ho_obj_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        self.union_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, self.act_repr_dim),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(0.5),
                                              nn.Linear(self.act_repr_dim, self.act_repr_dim),
                                              nn.ReLU(inplace=True),
                                              nn.Dropout(0.5),
                                              ])
        nn.init.xavier_normal_(self.union_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.union_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

        max_modes = 3
        self.act_output_mat = nn.Parameter(torch.empty((self.act_repr_dim, dataset.num_predicates, max_modes)), requires_grad=True)  # D x P x M
        torch.nn.init.xavier_normal_(self.act_output_mat, gain=1.0)

        # self.act_output_fc = nn.Linear(self.act_repr_dim, dataset.num_predicates, bias=False)
        # torch.nn.init.xavier_normal_(self.act_output_fc.weight, gain=1.0)

    @property
    def act_repr_dim(self):
        return self._act_repr_dim

    def _forward(self, vis_output: VisualOutput, **kwargs):
        if vis_output.box_labels is not None:
            vis_output.filter_boxes()
        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        masks = vis_output.masks
        union_boxes_feats = vis_output.hoi_union_boxes_feats
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

        box_feats_ext = torch.cat([box_feats, boxes_ext[:, 5:]], dim=1)

        ho_subj_repr = self.ho_subj_repr_mlp(box_feats_ext[hoi_infos[:, 1], :])
        ho_obj_repr = self.ho_obj_repr_mlp(box_feats_ext[hoi_infos[:, 2], :])
        union_repr = self.union_repr_mlp(union_boxes_feats)
        act_repr = union_repr + ho_subj_repr + ho_obj_repr

        action_logits = torch.einsum('nd,dpm->npm', (act_repr, self.act_output_mat))  # N x P x M
        assert action_logits.shape[0] == act_repr.shape[0] and \
               action_logits.shape[1] == self.dataset.num_predicates and \
               action_logits.shape[2] == self.act_output_mat.shape[2]
        action_logits = action_logits.max(dim=2)[0]  # N x P

        return action_logits


class GEmbModel(BaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'gemb'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)

        self.act_emb_att_fc = nn.Sequential(nn.Linear(self.act_repr_dim, dataset.num_predicates, bias=False),
                                            nn.Sigmoid())

        self.act_embs = nn.Parameter(torch.from_numpy(self.get_act_graph_embs()), requires_grad=False)
        self.act_only_repr_mlp = nn.Sequential(*[nn.Linear(self.act_repr_dim + self.act_embs.shape[1], self.act_repr_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(0.5),
                                                 nn.Linear(self.act_repr_dim, self.act_repr_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(0.5),
                                                 ])
        nn.init.xavier_normal_(self.act_only_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.act_only_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))

    def get_act_graph_embs(self):
        emb_path = 'cache/rotate_hico_act/'
        entity_embs = np.load(os.path.join(emb_path, 'entity_embedding.npy'))
        with open(os.path.join(emb_path, 'entities.dict'), 'r') as f:
            ecl_idx, entity_classes = zip(*[l.strip().split('\t') for l in f.readlines()])  # the index is loaded just for assertion check.
            ecl_idx = [int(x) for x in ecl_idx]
            assert np.all(np.arange(len(ecl_idx)) == np.array(ecl_idx))
            entity_inv_index = {e: i for i, e in enumerate(entity_classes)}
        act_embs = entity_embs[np.array([entity_inv_index[p] for p in self.dataset.predicates])]
        return act_embs

    def _forward(self, vis_output: VisualOutput, return_repr=False):
        if vis_output.box_labels is not None:
            vis_output.filter_boxes()
        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        masks = vis_output.masks
        union_boxes_feats = vis_output.hoi_union_boxes_feats
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

        box_feats_ext = torch.cat([box_feats, boxes_ext[:, 5:]], dim=1)

        ho_subj_repr = self.ho_subj_repr_mlp(box_feats_ext[hoi_infos[:, 1], :])
        ho_obj_repr = self.ho_obj_repr_mlp(box_feats_ext[hoi_infos[:, 2], :])
        union_repr = self.union_repr_mlp(union_boxes_feats)
        act_repr = union_repr + ho_subj_repr + ho_obj_repr
        if return_repr:
            return act_repr

        act_scores = self.act_emb_att_fc(act_repr)
        act_wavg_emb = act_scores @ self.act_embs
        act_emb_repr = self.act_only_repr_mlp(torch.cat([act_repr, act_wavg_emb], dim=1))
        action_logits = self.act_output_fc(act_emb_repr)

        return action_logits


class WEmbModel(GEmbModel):
    @classmethod
    def get_cline_name(cls):
        return 'wemb'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.word_emb_dim = 300

        word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        self.act_embs = nn.Parameter(torch.from_numpy(word_embs.get_embeddings(dataset.predicates)), requires_grad=False)
        self.act_only_repr_mlp = nn.Sequential(*[nn.Linear(self.act_repr_dim + self.act_embs.shape[1], self.act_repr_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(0.5),
                                                 nn.Linear(self.act_repr_dim, self.act_repr_dim),
                                                 nn.ReLU(inplace=True),
                                                 nn.Dropout(0.5),
                                                 ])
        nn.init.xavier_normal_(self.act_only_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('relu'))
        nn.init.xavier_normal_(self.act_only_repr_mlp[3].weight, gain=torch.nn.init.calculate_gain('relu'))


class ZSBaseModel(GenericModel):
    def __init__(self, dataset: HicoDetSplit, **kwargs):
        self.word_emb_dim = 300
        super().__init__(dataset, **kwargs)

        # FIXME
        base_model = BaseModel(dataset)
        if torch.cuda.is_available():
            base_model.cuda()
        ckpt = torch.load('output/base/2019-06-05_17-43-04_vanilla/final.tar')
        base_model.load_state_dict(ckpt['state_dict'])

        self.predictor_dim = base_model.act_repr_dim
        self.base_model = base_model
        self.normalize = cfg.model.wnorm
        if self.normalize:
            self.gt_classifiers = nn.functional.normalize(self.base_model.act_output_fc.weight.detach(), dim=1).unsqueeze(dim=0)  # 1 x P x D
        else:
            self.gt_classifiers = self.base_model.act_output_fc.weight.detach().unsqueeze(dim=0)  # 1 x P x D

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos is not None:
                hoi_predictors = self._forward(vis_output, )
            else:
                assert inference
                hoi_predictors = None

            if not inference:
                action_labels = vis_output.action_labels
                # obj_label_per_ho_pair = vis_output.box_labels[vis_output.ho_infos[:, 2]]
                # act_nz = torch.nonzero(action_labels)
                # hois = self.dataset.hicodet.op_pair_to_interaction[obj_label_per_ho_pair.cpu().numpy(), act_nz[:, 1].cpu().numpy()]
                # assert np.all(hois >= 0)
                # hoi_labels = action_labels.new_zeros((action_labels.shape[0], self.dataset.hicodet.num_interactions))
                # hoi_labels[act_nz[:, 0], torch.tensor(hois, device=hoi_labels.device)] = 1

                target_classifiers = action_labels.unsqueeze(dim=2) * self.gt_classifiers.expand(action_labels.shape[0], -1, -1)  # N x P x D
                losses = {'action_loss': nn.functional.mse_loss(action_labels.unsqueeze(dim=2) * hoi_predictors, target_classifiers, reduction='sum')}
                # losses = {'action_loss': nn.functional.binary_cross_entropy_with_logits(action_output, action_labels) * action_output.shape[1]}
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
                        assert hoi_predictors is not None
                        visual_feats = self.base_model._forward(vis_output, return_repr=True).detach().unsqueeze(dim=1)  # N x 1 x D
                        if self.normalize:
                            visual_feats = nn.functional.normalize(visual_feats, dim=2)
                        action_output = torch.bmm(visual_feats, hoi_predictors.transpose(1, 2)).squeeze(dim=1)

                        prediction.ho_img_inds = vis_output.ho_infos[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos[:, 1:]
                        # prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()
                        prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()  # For MSE

                return prediction


class ZSModel(ZSBaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'zs'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        self.word_emb_dim = 300
        super().__init__(dataset, **kwargs)
        self.dataset = dataset
        self.num_objects = dataset.num_object_classes
        self.num_predicates = dataset.num_predicates

        word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        obj_word_embs = word_embs.get_embeddings(dataset.objects, retry='avg_norm_last')
        pred_word_embs = word_embs.get_embeddings(dataset.predicates, retry='avg_norm_first')
        # self.obj_word_embs = nn.Embedding.from_pretrained(torch.from_numpy(obj_word_embs), freeze=True)
        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_word_embs), requires_grad=False)
        # self.pred_word_embs = nn.Embedding.from_pretrained(torch.from_numpy(pred_word_embs), freeze=True)
        self.pred_word_embs = nn.Parameter(torch.from_numpy(pred_word_embs), requires_grad=False)

        self.emb_to_predictor = nn.Sequential(*[nn.Linear(self.word_emb_dim * 2, self.predictor_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(self.predictor_dim, self.predictor_dim),
                                                ])

    def _forward(self, vis_output: VisualOutput, **kwargs):
        if vis_output.box_labels is not None:
            vis_output.filter_boxes()
        boxes_ext = vis_output.boxes_ext
        masks = vis_output.masks
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

        if vis_output.box_labels is not None:
            obj_word_embs = self.obj_word_embs[vis_output.box_labels][hoi_infos[:, 2]]
        else:
            obj_word_embs = self.obj_word_embs[boxes_ext.argmax(dim=1)][hoi_infos[:, 2]]
        batch_size = hoi_infos.shape[0]
        num_preds = self.pred_word_embs.shape[0]
        emb_dim = self.word_emb_dim
        hoi_word_embs = torch.cat([obj_word_embs.unsqueeze(dim=1).expand(batch_size, num_preds, emb_dim),
                                   self.pred_word_embs.unsqueeze(dim=0).expand(batch_size, num_preds, emb_dim)
                                   ], dim=2)  # N x P x 2*E
        if self.normalize:
            hoi_predictors = nn.functional.normalize(self.emb_to_predictor(hoi_word_embs), dim=2)  # N x P x D
        else:
            hoi_predictors = self.emb_to_predictor(hoi_word_embs)  # N x P x D
        return hoi_predictors


class ZSVModel(ZSBaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'zsv'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        self.word_emb_dim = 300
        super().__init__(dataset, **kwargs)
        self.dataset = dataset
        self.num_objects = dataset.num_object_classes
        self.num_predicates = dataset.num_predicates

        word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        obj_word_embs = word_embs.get_embeddings(dataset.objects, retry='avg_norm_last')
        pred_word_embs = word_embs.get_embeddings(dataset.predicates, retry='avg_norm_first')
        # self.obj_word_embs = nn.Embedding.from_pretrained(torch.from_numpy(obj_word_embs), freeze=True)
        self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_word_embs), requires_grad=False)
        # self.pred_word_embs = nn.Embedding.from_pretrained(torch.from_numpy(pred_word_embs), freeze=True)
        self.pred_word_embs = nn.Parameter(torch.from_numpy(pred_word_embs), requires_grad=False)

        self.emb_to_predictor = nn.Sequential(*[nn.Linear(self.word_emb_dim * 2 + self.predictor_dim, self.predictor_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(self.predictor_dim, self.predictor_dim),
                                                # nn.ReLU(inplace=True),
                                                # nn.Dropout(0.5),
                                                # nn.Linear(self.predictor_dim, self.predictor_dim),
                                                ])

    def _forward(self, vis_output: VisualOutput, **kwargs):
        if vis_output.box_labels is not None:
            vis_output.filter_boxes()
        boxes_ext = vis_output.boxes_ext
        masks = vis_output.masks
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

        if vis_output.box_labels is not None:
            obj_word_embs = self.obj_word_embs[vis_output.box_labels][hoi_infos[:, 2]]
        else:
            obj_word_embs = self.obj_word_embs[boxes_ext.argmax(dim=1)][hoi_infos[:, 2]]

        batch_size = hoi_infos.shape[0]
        num_preds = self.pred_word_embs.shape[0]

        vrepr = self.base_model._forward(vis_output, return_repr=True).detach()  # N x D
        repr = torch.cat([vrepr.unsqueeze(dim=1).expand(-1, num_preds, -1),
                          obj_word_embs.unsqueeze(dim=1).expand(-1, num_preds, -1),
                          self.pred_word_embs.unsqueeze(dim=0).expand(batch_size, -1, -1)
                          ], dim=2)  # N x P x (2*E + D)

        if self.normalize:
            hoi_predictors = nn.functional.normalize(self.emb_to_predictor(repr), dim=2)  # N x P x D
        else:
            hoi_predictors = self.emb_to_predictor(repr)  # N x P x D
        return hoi_predictors


class ZSAutoencoderModel(ZSBaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'zsae'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        self.word_emb_dim = 300
        super().__init__(dataset, **kwargs)
        self.dataset = dataset
        self.num_objects = dataset.num_object_classes
        self.num_predicates = dataset.num_predicates
        assert self.normalize

        word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim, normalize=self.normalize)
        # obj_word_embs = word_embs.get_embeddings(dataset.objects, retry='avg_norm_last')
        # self.obj_word_embs = nn.Parameter(torch.from_numpy(obj_word_embs), requires_grad=False)

        pred_word_embs = word_embs.get_embeddings(dataset.predicates, retry='avg_norm_first')
        # self.pred_word_embs = nn.Embedding.from_pretrained(torch.from_numpy(pred_word_embs), freeze=True)
        # self.pred_word_embs = nn.Parameter(torch.from_numpy(pred_word_embs), requires_grad=False)  # P x E
        self.pred_word_embs = nn.Parameter(torch.from_numpy(pred_word_embs.T), requires_grad=False)  # E x P

        self.vrepr_to_emb = nn.Sequential(*[nn.Linear(self.predictor_dim, self.predictor_dim),
                                            nn.ReLU(inplace=True),
                                            nn.Dropout(0.5),
                                            nn.Linear(self.predictor_dim, self.word_emb_dim),
                                            ])
        self.emb_to_predictor = nn.Sequential(*[nn.Linear(self.word_emb_dim, self.predictor_dim),
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                nn.Linear(self.predictor_dim, self.predictor_dim),
                                                ])

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos is not None:
                act_embeddings, act_predictors, vrepr = self._forward(vis_output)
            else:
                assert inference
                act_embeddings = act_predictors = None

            if not inference:
                act_labels = vis_output.action_labels
                # obj_label_per_ho_pair = vis_output.box_labels[vis_output.ho_infos[:, 2]]
                # act_nz = torch.nonzero(act_labels)
                # hois = self.dataset.hicodet.op_pair_to_interaction[obj_label_per_ho_pair.cpu().numpy(), act_nz[:, 1].cpu().numpy()]
                # assert np.all(hois >= 0)
                # hoi_labels = act_labels.new_zeros((act_labels.shape[0], self.dataset.hicodet.num_interactions))
                # hoi_labels[act_nz[:, 0], torch.tensor(hois, device=hoi_labels.device)] = 1

                act_embeddings = act_embeddings.transpose(2, 1)
                target_embeddings = self.pred_word_embs.unsqueeze(dim=0).expand(act_labels.shape[0], -1, -1)  # N x E x P
                # emb_distances = ((act_embeddings - target_embeddings) ** 2).sum(dim=2)

                target_classifiers = self.gt_classifiers.expand(act_labels.shape[0], -1, -1)  # N x P x D
                # cl_distances = ((act_predictors - target_classifiers) ** 2).sum(dim=2)

                losses = {'a2emb_loss': nn.functional.cosine_embedding_loss(act_embeddings, target_embeddings, 2 * act_labels - 1, reduction='sum'),
                          # 'emb2cl_loss': nn.functional.multilabel_margin_loss(cl_distances, al_long, reduction='sum'),
                          'emb2cl_loss': nn.functional.mse_loss(act_labels.unsqueeze(dim=2) * act_predictors,
                                                                act_labels.unsqueeze(dim=2) * target_classifiers, reduction='sum')
                          }
                # losses = {'action_loss': nn.functional.binary_cross_entropy_with_logits(action_output, act_labels) * action_output.shape[1]}
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
                        assert act_predictors is not None

                        action_output = torch.bmm(vrepr, act_predictors.transpose(1, 2)).squeeze(dim=1)

                        prediction.ho_img_inds = vis_output.ho_infos[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos[:, 1:]
                        # prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()
                        prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()  # For MSE

                return prediction

    def _forward(self, vis_output: VisualOutput, **kwargs):
        if vis_output.box_labels is not None:
            vis_output.filter_boxes()
        # boxes_ext = vis_output.boxes_ext
        # masks = vis_output.masks
        # hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)
        #
        # if vis_output.box_labels is not None:
        #     obj_word_embs = self.obj_word_embs[vis_output.box_labels][hoi_infos[:, 2]]
        # else:
        #     obj_word_embs = self.obj_word_embs[boxes_ext.argmax(dim=1)][hoi_infos[:, 2]]
        # batch_size = hoi_infos.shape[0]
        # num_preds = self.pred_word_embs.shape[0]
        # emb_dim = self.word_emb_dim
        # hoi_word_embs = torch.cat([obj_word_embs.unsqueeze(dim=1).expand(batch_size, num_preds, emb_dim),
        #                            self.pred_word_embs.unsqueeze(dim=0).expand(batch_size, num_preds, emb_dim)
        #                            ], dim=2)  # N x P x 2*E

        vrepr = self.base_model._forward(vis_output, return_repr=True).detach()
        act_emb = self.vrepr_to_emb(vrepr).unsqueeze(dim=1).expand(-1, self.num_predicates, -1)  # N x P x E
        # if vis_output.action_labels is not None:
        #     target_embs = self.pred_word_embs.unsqueeze(dim=0).expand(vrepr.shape[0], -1, -1) * vis_output.action_labels.unsqueeze(dim=2)
        #     act_predictors = self.emb_to_predictor(target_embs)  # N x P x D
        # else:
        act_predictors = self.emb_to_predictor(act_emb.detach())  # N x P x D
        vrepr = vrepr.unsqueeze(dim=1)  # N x 1 x D
        if self.normalize:
            act_emb = nn.functional.normalize(act_emb, dim=2)
            act_predictors = nn.functional.normalize(act_predictors, dim=2)
            vrepr = nn.functional.normalize(vrepr, dim=2)
        return act_emb, act_predictors, vrepr


class PeyreModel(GenericModel):
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
                hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits = self._forward(vis_output, )
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

    def _forward(self, vis_output: VisualOutput, **kwargs):
        if vis_output.box_labels is not None:
            vis_output.filter_boxes()
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
