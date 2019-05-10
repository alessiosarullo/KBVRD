from lib.bbox_utils import compute_ious
from lib.dataset.utils import Splits
from lib.models.generic_model import GenericModel, Prediction, Minibatch
from lib.models.hoi_branches import *
from lib.models.obj_branches import *


class OracleModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'oracle'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.fake = torch.nn.Parameter(torch.from_numpy(np.array([1.])), requires_grad=True)
        self.iou_thresh = 0.5
        self.split = Splits.TEST
        self.perfect_detector = True

    def forward(self, x: Minibatch, inference=True, **kwargs):
        assert inference is True
        assert not self.training

        boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, hoi_infos, box_labels, action_labels, hoi_labels = self.visual_module(x, True)
        im_scales = x.img_infos[:, 2].cpu().numpy()
        gt_entry = HicoDetInstanceSplit.get_split(self.split).get_entry(x.other_ex_data[0]['index'], read_img=False, ignore_precomputed=True)
        gt_boxes = gt_entry.gt_boxes * im_scales[0]
        gt_obj_classes = gt_entry.gt_obj_classes
        if self.perfect_detector:
            boxes_ext = torch.from_numpy(np.concatenate([np.zeros((gt_boxes.shape[0], 1)),
                                                         gt_boxes,
                                                         self.visual_module.one_hot_obj_labels(gt_obj_classes)
                                                         ], axis=1))
            hoi_infos = self.visual_module.get_all_pairs(boxes_ext[:, :5].detach().cpu().numpy(), gt_obj_classes)
            if hoi_infos.size == 0:
                hoi_infos = None

        if hoi_infos is not None:
            im_ids = np.unique(hoi_infos[:, 0])
            assert im_ids.size == 1 and im_ids == 0

            predict_boxes = boxes_ext[:, 1:5].detach().cpu().numpy()
            pred_gt_ious = compute_ious(predict_boxes, gt_boxes)

            pred_gt_best_match = np.argmax(pred_gt_ious, axis=1)  # type: np.ndarray
            box_labels = gt_obj_classes[pred_gt_best_match]  # assign the best match
            obj_output = torch.from_numpy(self.visual_module.one_hot_obj_labels(box_labels))

            pred_gt_ious_class_match = (box_labels[:, None] == gt_obj_classes[None, :])

            predict_ho_pairs = hoi_infos[:, 1:]
            gt_hois = gt_entry.gt_hois[:, [0, 2, 1]]

            action_labels = np.zeros((predict_ho_pairs.shape[0], self.dataset.num_predicates))
            for predict_idx, (ph, po) in enumerate(predict_ho_pairs):
                gt_pair_ious = np.zeros(gt_hois.shape[0])
                for gtidx, (gh, go, gi) in enumerate(gt_hois):
                    iou_h = pred_gt_ious[ph, gh]
                    iou_o = pred_gt_ious[po, go]
                    if pred_gt_ious_class_match[ph, gh] and pred_gt_ious_class_match[po, go]:
                        gt_pair_ious[gtidx] = min(iou_h, iou_o)
                if np.any(gt_pair_ious > self.iou_thresh):
                    gtidxs = (gt_pair_ious > self.iou_thresh)
                    action_labels[predict_idx, np.unique(gt_hois[gtidxs, 2])] = 1

            action_output = torch.from_numpy(action_labels)
        else:
            obj_output = action_output = boxes_ext = None

        if not inference:
            assert obj_output is not None and action_output is not None and box_labels is not None and action_labels is not None
            return obj_output, action_output, box_labels, action_labels
        else:
            return self._prepare_prediction(obj_output, action_output, hoi_infos, boxes_ext, im_scales=im_scales)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        raise NotImplementedError()

    @staticmethod
    def _prepare_prediction(obj_output, action_output, hoi_output, hoi_infos, boxes_ext, im_scales):
        if hoi_infos is not None:
            assert obj_output is not None and action_output is not None and boxes_ext is not None
            obj_prob = obj_output.cpu().numpy()
            action_probs = action_output.cpu().numpy()
            ho_img_inds = hoi_infos[:, 0]
            ho_pairs = hoi_infos[:, 1:]
        else:
            action_probs = ho_pairs = ho_img_inds = None
            obj_prob = None

        if boxes_ext is not None:
            boxes_ext = boxes_ext.cpu().numpy()
            obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
            obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
            if obj_prob is None:
                obj_prob = boxes_ext[:, 5:]  # this cannot be refined because of the lack of spatial relationships
        else:
            obj_im_inds = obj_boxes = None
        return Prediction(obj_im_inds=obj_im_inds,
                          obj_boxes=obj_boxes,
                          obj_scores=obj_prob,
                          ho_img_inds=ho_img_inds,
                          ho_pairs=ho_pairs,
                          action_scores=action_probs)


class ObjectOnlyModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        raise NotImplementedError()

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)

    def forward(self, x: Minibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, hoi_infos, box_labels, _, _ = \
                self.visual_module(x, inference)
            # `hoi_infos` is an R x 3 NumPy array where each column is [image ID, subject index, object index].
            # Masks are floats at this point.

            obj_output = action_output = hoi_output = None
            if boxes_ext is not None:
                obj_output, _, _ = self._forward(boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels)

            if not inference:
                assert box_labels is not None
                return obj_output, None, None, box_labels, None, None
            else:
                im_scales = x.img_infos[:, 2].cpu().numpy()
                return self._prepare_prediction(obj_output, action_output, hoi_output, hoi_infos, boxes_ext, im_scales)

    @staticmethod
    def _prepare_prediction(obj_output, action_output, hoi_output, hoi_infos, boxes_ext, im_scales):
        if boxes_ext is not None:
            assert obj_output is not None
            obj_prob = nn.functional.softmax(obj_output, dim=1).cpu().numpy()
            boxes_ext = boxes_ext.cpu().numpy()
            obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
            obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
        else:
            obj_prob = obj_im_inds = obj_boxes = None
        return Prediction(obj_im_inds=obj_im_inds,
                          obj_boxes=obj_boxes,
                          obj_scores=obj_prob,
                          ho_img_inds=None,
                          ho_pairs=None,
                          action_scores=None,
                          hoi_scores=None)


class ObjectOnlyZeroModel(ObjectOnlyModel):
    @classmethod
    def get_cline_name(cls):
        return 'objonlyzero'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.fake = torch.nn.Parameter(torch.from_numpy(np.array([1.])), requires_grad=True)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        obj_logits = boxes_ext[:, 5:]
        return obj_logits, None, None


class ObjectOnlyVisModel(ObjectOnlyModel):
    @classmethod
    def get_cline_name(cls):
        return 'objonlyvis'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.obj_output_fc = nn.Linear(self.visual_module.vis_feat_dim, self.dataset.num_object_classes)
        torch.nn.init.xavier_normal_(self.obj_output_fc.weight, gain=1.0)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        obj_logits = self.obj_output_fc(box_feats)
        return obj_logits, None, None


class ObjectOnlyEmbModel(ObjectOnlyModel):
    @classmethod
    def get_cline_name(cls):
        raise NotImplementedError()

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.use_gt = False

    def forward(self, x: Minibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            im_scales = x.img_infos[:, 2].cpu().numpy()
            if self.use_gt:
                boxes_ext = []
                gt_obj_classes = []
                for i, data in enumerate(x.other_ex_data):
                    gt_entry = HicoDetInstanceSplit.get_split(data['split']).get_entry(data['index'], read_img=False, ignore_precomputed=True)
                    gt_obj_classes.append(gt_entry.gt_obj_classes)
                    gt_boxes = gt_entry.gt_boxes * im_scales[i]
                    boxes_ext.append(np.concatenate([np.zeros((gt_boxes.shape[0], 1)),
                                                     gt_boxes,
                                                     self.visual_module.one_hot_obj_labels(gt_entry.gt_obj_classes)
                                                     ], axis=1))
                box_labels = torch.tensor(np.concatenate(gt_obj_classes, axis=0), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
                boxes_ext = torch.tensor(np.concatenate(boxes_ext, axis=0), device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            else:
                boxes_ext, _, _, _, _, _, box_labels, _, _ = self.visual_module(x, inference)

            obj_output, _, _ = self._forward(boxes_ext=boxes_ext, box_feats=None, masks=None, union_boxes_feats=None, hoi_infos=None, box_labels=None)

            if not inference:
                assert box_labels is not None
                return obj_output, None, None, box_labels, None, None
            else:
                return self._prepare_prediction(obj_output, None, None, None, boxes_ext, im_scales)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        obj_logits = self.obj_output_fc(self.obj_embs[torch.argmax(boxes_ext[:, 5:], dim=1), :])
        return obj_logits, None, None


class ObjectOnlyRotateEmbModel(ObjectOnlyEmbModel):
    @classmethod
    def get_cline_name(cls):
        return 'objonlyemb'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        entity_embs = np.load('cache/rotate/entity_embedding.npy')
        with open('cache/rotate/entities.dict', 'r') as f:
            ecl_idx, entity_classes = zip(*[l.strip().split('\t') for l in f.readlines()])  # the index is loaded just for assertion check.
            ecl_idx = [int(x) for x in ecl_idx]
            assert np.all(np.arange(len(ecl_idx)) == np.array(ecl_idx))
            entity_inv_index = {e: i for i, e in enumerate(entity_classes)}
        obj_inds = np.array([entity_inv_index[o] for o in dataset.objects])
        obj_embs = entity_embs[obj_inds]

        self.obj_embs = nn.Parameter(torch.from_numpy(obj_embs), requires_grad=False)
        self.obj_output_fc = nn.Linear(obj_embs.shape[1], self.dataset.num_object_classes)
        torch.nn.init.xavier_normal_(self.obj_output_fc.weight, gain=1.0)


class ObjectOnlyWordEmbModel(ObjectOnlyEmbModel):
    @classmethod
    def get_cline_name(cls):
        return 'objonlywordemb'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        self.word_emb_dim = 300
        super().__init__(dataset, **kwargs)
        word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        obj_embs = word_embs.get_embeddings([o for o in dataset.objects])

        self.obj_embs = nn.Parameter(torch.from_numpy(obj_embs), requires_grad=False)
        self.obj_output_fc = nn.Linear(obj_embs.shape[1], self.dataset.num_object_classes)
        torch.nn.init.xavier_normal_(self.obj_output_fc.weight, gain=1.0)


class ObjectOnlyWordEmb7Model(ObjectOnlyEmbModel):
    @classmethod
    def get_cline_name(cls):
        return 'objonlywordemb7'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        self.word_emb_dim = 300
        super().__init__(dataset, **kwargs)
        word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
        obj_embs = word_embs.get_embeddings([o for o in dataset.objects])

        self.obj_embs = nn.Parameter(torch.from_numpy(np.tile(obj_embs, reps=[1, 7])), requires_grad=False)
        self.obj_output_fc = nn.Linear(self.obj_embs.shape[1], self.dataset.num_object_classes)
        torch.nn.init.xavier_normal_(self.obj_output_fc.weight, gain=1.0)


class ActionOnlyGOEmbModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'gobjemb'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = EmbObjBranch(dataset=dataset, vis_dim=vis_feat_dim)
        self.act_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_dim)

        self.obj_output_fc = nn.Linear(self.obj_branch.output_dim, self.dataset.num_object_classes)
        self.action_output_fc = nn.Linear(self.act_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.action_output_fc.weight, gain=1.0)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids)
        obj_logits = self.obj_output_fc(obj_repr)

        hoi_repr = self.act_branch(obj_repr, union_boxes_feats, hoi_infos)
        action_logits = self.action_output_fc(hoi_repr)

        return obj_logits, action_logits, None


class ActionOnlyModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'actonly'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = SimpleObjBranch(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.act_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_dim)

        self.obj_output_fc = nn.Linear(self.obj_branch.output_dim, self.dataset.num_object_classes)
        self.action_output_fc = nn.Linear(self.act_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.action_output_fc.weight, gain=1.0)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids)
        obj_logits = self.obj_output_fc(obj_repr)

        act_repr = self.act_branch(obj_repr, union_boxes_feats, hoi_infos)
        action_logits = self.action_output_fc(act_repr)

        return obj_logits, action_logits, None


class HoiOnlyModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'hoionly'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = SimpleObjBranch(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.hoi_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_dim, hoi_repr_dim=1024)  # FIXME magic constant

        self.obj_output_fc = nn.Linear(self.obj_branch.output_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_interactions, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids)
        obj_logits = self.obj_output_fc(obj_repr)

        hoi_repr = self.hoi_branch(obj_repr, union_boxes_feats, hoi_infos)
        hoi_logits = self.hoi_output_fc(hoi_repr)

        return obj_logits, None, hoi_logits


class HoiModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'hoi'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = SimpleObjBranch(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.act_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_dim)  # FIXME magic constant
        self.hoi_branch = SimpleHoiBranch(self.act_branch.output_dim, self.obj_branch.output_dim, hoi_repr_dim=1024)  # FIXME magic constant

        self.obj_output_fc = nn.Linear(self.obj_branch.output_dim, self.dataset.num_object_classes)
        self.act_output_fc = nn.Linear(self.act_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.act_output_fc.weight, gain=1.0)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_interactions, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids)
        obj_logits = self.obj_output_fc(obj_repr)

        act_repr = self.act_branch(obj_repr, union_boxes_feats, hoi_infos)
        act_logits = self.act_output_fc(act_repr)

        hoi_repr = self.hoi_branch(obj_repr, act_repr, hoi_infos)
        hoi_logits = self.hoi_output_fc(hoi_repr)

        return obj_logits, act_logits, hoi_logits


class EmbsimActPredModel(ActionOnlyModel):
    @classmethod
    def get_cline_name(cls):
        return 'embsimactpred'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = SimpleObjBranch(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.act_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.repr_dim)  # FIXME magic constant

        self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)
        self.act_output_fc = nn.Linear(self.act_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.act_output_fc.weight, gain=1.0)

        self.act_embsim_branch = HoiEmbsimBranch(self.act_branch.output_dim, self.obj_branch.repr_dim, dataset)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids)
        obj_logits = self.obj_output_fc(obj_repr)

        act_repr = self.act_branch(obj_repr, union_boxes_feats, hoi_infos)
        act_logits = self.act_output_fc(act_repr)

        emb_act_logits = self.act_embsim_branch(act_repr, obj_repr, hoi_infos)
        act_logits = act_logits + emb_act_logits

        return obj_logits, act_logits, None


class KatoModel(GenericModel):
    # FIXME?
    @classmethod
    def get_cline_name(cls):
        return 'kato'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.obj_branch = SimpleObjBranch(input_dim=self.visual_module.vis_feat_dim + self.dataset.num_object_classes)
        self.obj_output_fc = nn.Linear(self.obj_branch.output_dim, self.dataset.num_object_classes)

        self.hoi_branch = KatoGCNBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_dim, dataset)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids)
        obj_logits = self.obj_output_fc(obj_repr)

        hoi_obj_logits, action_logits = self.hoi_branch(obj_repr, union_boxes_feats, hoi_infos)

        return obj_logits, action_logits


class PeyreModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'peyre'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.hoi_branch = PeyreEmbsimBranch(self.visual_module.vis_feat_dim, dataset)

    def get_losses(self, x, **kwargs):
        y = self(x, inference=False, **kwargs)

        hoi_subj_logits, hoi_subj_labels = y[0]
        subj_labels_1hot = hoi_subj_labels.new_zeros((hoi_subj_labels.shape[0], self.dataset.num_object_classes)).float()
        subj_labels_1hot[torch.arange(subj_labels_1hot.shape[0]), hoi_subj_labels] = 1
        hoi_subj_loss = nn.functional.binary_cross_entropy_with_logits(hoi_subj_logits, subj_labels_1hot) * self.dataset.num_object_classes

        hoi_obj_logits, hoi_obj_labels = y[1]
        obj_labels_1hot = hoi_obj_labels.new_zeros((hoi_obj_labels.shape[0], self.dataset.num_object_classes)).float()
        obj_labels_1hot[torch.arange(obj_labels_1hot.shape[0]), hoi_obj_labels] = 1
        hoi_obj_loss = nn.functional.binary_cross_entropy_with_logits(hoi_obj_logits, obj_labels_1hot) * self.dataset.num_object_classes

        hoi_act_logits, action_labels = y[2]
        act_loss = nn.functional.binary_cross_entropy_with_logits(hoi_act_logits, action_labels) * self.dataset.num_predicates

        hoi_logits, hoi_labels = y[3]
        hoi_loss = nn.functional.binary_cross_entropy_with_logits(hoi_logits, hoi_labels) * self.dataset.num_interactions

        return {'hoi_subj_loss': hoi_subj_loss, 'hoi_obj_loss': hoi_obj_loss, 'action_loss': act_loss, 'hoi_loss': hoi_loss}

    def forward(self, x, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, hoi_infos, box_labels, action_labels, hoi_labels = \
                self.visual_module(x, inference)
            # `hoi_infos` is an R x 3 NumPy array where each column is [image ID, subject index, object index].
            # Masks are floats at this point.

            if hoi_infos is not None:
                logits = self._forward(boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels, action_labels)
                hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits = logits
            else:
                hoi_subj_logits = hoi_obj_logits = hoi_act_logits = hoi_logits = None

            if not inference:
                assert all([x is not None for x in (box_labels, action_labels, hoi_labels)])

                hoi_subj_labels = box_labels[hoi_infos[:, 1]]
                # assert torch.unique(hoi_subj_labels).shape[0] == 1, hoi_subj_labels  # Doesn't work because of negative sampling. FIXME?

                hoi_obj_labels = box_labels[hoi_infos[:, 2]]
                return (hoi_subj_logits, hoi_subj_labels), (hoi_obj_logits, hoi_obj_labels), (hoi_act_logits, action_labels), (hoi_logits, hoi_labels)
            else:
                im_scales = x.img_infos[:, 2].cpu().numpy()
                if hoi_logits is None:
                    hoi_output = None
                else:
                    hoi_output = np.empty([hoi_logits.shape[0], self.dataset.num_interactions])
                    for iid, (pid, oid) in enumerate(self.dataset.interactions):
                        hoi_subj_prob = torch.sigmoid(hoi_subj_logits[:, self.dataset.human_class])
                        hoi_obj_prob = torch.sigmoid(hoi_obj_logits[:, oid])
                        hoi_act_prob = torch.sigmoid(hoi_act_logits[:, pid])
                        hoi_prob = torch.sigmoid(hoi_logits[:, iid])
                        hoi_output[:, iid] = hoi_subj_prob * hoi_obj_prob * hoi_act_prob * hoi_prob
                return self._prepare_prediction(None, None, hoi_output, hoi_infos, boxes_ext, im_scales)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits = self.hoi_branch(boxes_ext, box_feats, hoi_infos)
        return hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits

    @staticmethod
    def _prepare_prediction(obj_output, action_output, hoi_probs, hoi_infos, boxes_ext, im_scales):
        assert obj_output is None and action_output is None

        ho_pairs = ho_img_inds = None
        if hoi_infos is not None:
            assert boxes_ext is not None
            assert hoi_probs is not None
            ho_img_inds = hoi_infos[:, 0]
            ho_pairs = hoi_infos[:, 1:]

        if boxes_ext is not None:
            boxes_ext = boxes_ext.cpu().numpy()
            obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
            obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
        else:
            obj_im_inds = obj_boxes = None
        return Prediction(obj_im_inds=obj_im_inds,
                          obj_boxes=obj_boxes,
                          obj_scores=None,
                          ho_img_inds=ho_img_inds,
                          ho_pairs=ho_pairs,
                          action_scores=None,
                          hoi_scores=hoi_probs)


def main():
    from lib.dataset.hicodet import Splits
    from scripts.utils import get_all_models_by_name

    cfg.parse_args(allow_required=False)
    hdtrain = HicoDetInstanceSplit.get_split(split=Splits.TRAIN)
    detector = get_all_models_by_name()[cfg.program.model](hdtrain)
    detector.cuda()


if __name__ == '__main__':
    main()
