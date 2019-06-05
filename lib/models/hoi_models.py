from lib.dataset.hicodet.pc_hicodet_split import PrecomputedMinibatch
from lib.models.containers import VisualOutput
from lib.models.generic_model import GenericModel, Prediction
from lib.models.hoi_branches import *
from lib.models.obj_branches import *


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


class CosModel(BaseModel):
    @classmethod
    def get_cline_name(cls):
        return 'cos'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        act_weights = torch.empty((self.act_repr_dim, dataset.num_predicates))
        torch.nn.init.xavier_normal_(act_weights, gain=1.0)
        self.act_weights = nn.Parameter(act_weights, requires_grad=True)
        self.act_output_cs = nn.CosineSimilarity(dim=1)

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

        action_logits = self.act_output_cs(act_repr.unsqueeze(dim=2), self.act_weights.unsqueeze(dim=0))
        return action_logits


class ZSModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'zs'

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        self.word_emb_dim = 300
        super().__init__(dataset, **kwargs)

        # FIXME
        base_model = BaseModel(dataset)
        if torch.cuda.is_available():
            base_model.cuda()
        ckpt = torch.load('output/base/2019-06-05_17-43-04_nobias/final.tar')
        base_model.load_state_dict(ckpt['state_dict'])

        self.predictor_dim = base_model.act_repr_dim

        self.dataset = dataset
        self.num_objects = dataset.num_object_classes
        self.num_predicates = dataset.num_predicates
        self.base_model = base_model

        if cfg.model.wnorm:
            self.gt_classifiers = nn.functional.normalize(self.base_model.act_output_fc.weight.detach(), dim=1).unsqueeze(dim=0)  # 1 x P x D
        else:
            self.gt_classifiers = self.base_model.act_output_fc.weight.detach().unsqueeze(dim=0)  # 1 x P x D

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
                                                nn.ReLU(inplace=True),
                                                nn.Dropout(0.5),
                                                ])

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
                        if cfg.model.wnorm:
                            visual_feats = nn.functional.normalize(visual_feats, dim=2)
                        action_output = torch.bmm(visual_feats, hoi_predictors.transpose(1, 2)).squeeze(dim=1)

                        prediction.ho_img_inds = vis_output.ho_infos[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos[:, 1:]
                        # prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()
                        prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()  # For MSE

                return prediction

    def _forward(self, vis_output: VisualOutput, **kwargs):
        if vis_output.box_labels is not None:
            vis_output.filter_boxes()
        boxes_ext = vis_output.boxes_ext
        masks = vis_output.masks
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

        obj_word_embs = boxes_ext[hoi_infos[:, 2], 5:] @ self.obj_word_embs
        batch_size = hoi_infos.shape[0]
        num_preds = self.pred_word_embs.shape[0]
        emb_dim = self.word_emb_dim
        hoi_word_embs = torch.cat([obj_word_embs.unsqueeze(dim=1).expand(batch_size, num_preds, emb_dim),
                                   self.pred_word_embs.unsqueeze(dim=0).expand(batch_size, num_preds, emb_dim)
                                   ], dim=2)  # N x P x 2*E
        if cfg.model.wnorm:
            hoi_predictors = nn.functional.normalize(self.emb_to_predictor(hoi_word_embs), dim=2)  # N x P x D
        else:
            hoi_predictors = self.emb_to_predictor(hoi_word_embs)  # N x P x D
        return hoi_predictors
