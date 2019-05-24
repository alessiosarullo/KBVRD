from lib.bbox_utils import compute_ious
from lib.dataset.utils import Splits
from lib.models.generic_model import GenericModel, Prediction, Minibatch
from lib.models.containers import VisualOutput
from lib.models.hoi_branches import *
from lib.models.obj_branches import *


class OracleModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'oracle'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        raise NotImplementedError('This needs to be checked after refactors.')
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

    def _forward(self):
        raise NotImplementedError()

    def _prepare_prediction(self, obj_output, action_output, hoi_output, hoi_infos, boxes_ext, im_scales):
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


class ObjFGPredModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'objfgpred'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.box_thr = cfg.model.proposal_thr

        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = SimpleObjBranch(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.act_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_dim, use_relu=cfg.model.relu)

        self.obj_output_fc = nn.Linear(self.obj_branch.output_dim, self.dataset.num_object_classes)
        self.action_output_fc = nn.Linear(self.act_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.action_output_fc.weight, gain=1.0)

    def forward(self, x: Minibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos is not None:
                obj_output, action_output = self._forward(vis_output)
            else:
                obj_output = action_output = None

            if not inference:
                box_labels = vis_output.box_labels
                action_labels = vis_output.action_labels

                fg_box_inds = (box_labels >= 0)
                obj_output = obj_output[fg_box_inds, :]
                box_labels = box_labels[fg_box_inds]

                losses = {'object_loss': nn.functional.cross_entropy(obj_output, box_labels),
                          'action_loss': nn.functional.binary_cross_entropy_with_logits(action_output, action_labels) * action_output.shape[1]}
                return losses
            else:
                prediction = Prediction()

                if vis_output.ho_infos is not None:
                    assert action_output is not None

                    prediction.ho_img_inds = vis_output.ho_infos[:, 0]
                    prediction.ho_pairs = vis_output.ho_infos[:, 1:]
                    prediction.obj_prob = nn.functional.softmax(obj_output, dim=1).cpu().numpy()
                    prediction.action_probs = torch.sigmoid(action_output).cpu().numpy()

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2].cpu().numpy()

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes

                return prediction

    def _forward(self, vis_output: VisualOutput):
        if vis_output.box_labels is not None:
            bg_boxes, bg_box_feats, bg_masks = vis_output.filter_boxes(thr=None)

        if self.box_thr > 0:
            bg_boxes, bg_box_feats, bg_masks = vis_output.filter_boxes(thr=self.box_thr)
        if vis_output.ho_infos is None:
            return None, None

        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        masks = vis_output.masks
        union_boxes_feats = vis_output.hoi_union_boxes_feats
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

        obj_repr = self.obj_branch(boxes_ext, box_feats)
        obj_logits = self.obj_output_fc(obj_repr)

        act_repr = self.act_branch(obj_repr, union_boxes_feats, hoi_infos)
        action_logits = self.action_output_fc(act_repr)

        return obj_logits, action_logits


class ActionOnlyModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'actonly'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = SimpleObjBranch(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.act_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_dim, use_relu=cfg.model.relu)

        self.obj_output_fc = nn.Linear(self.obj_branch.output_dim, self.dataset.num_object_classes)
        self.action_output_fc = nn.Linear(self.act_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.action_output_fc.weight, gain=1.0)

    def _forward(self, vis_output: VisualOutput):
        if vis_output.box_labels is not None:
            vis_output.filter_boxes(thr=None)
        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        masks = vis_output.masks
        union_boxes_feats = vis_output.hoi_union_boxes_feats
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

        obj_repr = self.obj_branch(boxes_ext, box_feats)
        obj_logits = self.obj_output_fc(obj_repr)

        act_repr = self.act_branch(obj_repr, union_boxes_feats, hoi_infos)
        action_logits = self.action_output_fc(act_repr)

        return obj_logits, action_logits


class ActionOnlyHoiModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'hoiactonly'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = SimpleObjBranch(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.act_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_dim)

        self.obj_output_fc = nn.Linear(self.obj_branch.output_dim, self.dataset.num_object_classes)
        self.action_output_fc = nn.Linear(self.act_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.action_output_fc.weight, gain=1.0)

    def forward(self, x: Minibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos is not None:
                obj_output, action_output = self._forward(vis_output)
            else:
                obj_output = action_output = None

            if not inference:
                box_labels = vis_output.box_labels
                action_labels = vis_output.action_labels

                fg_box_inds = (box_labels >= 0)
                obj_output = obj_output[fg_box_inds, :]
                box_labels = box_labels[fg_box_inds]

                losses = {'object_loss': nn.functional.cross_entropy(obj_output, box_labels),
                          'action_loss': nn.functional.binary_cross_entropy_with_logits(action_output, action_labels) * action_output.shape[1]}
                return losses
            else:
                prediction = Prediction()

                if vis_output.ho_infos is not None:
                    assert action_output is not None

                    ho_infos = vis_output.ho_infos
                    obj_prob = nn.functional.softmax(obj_output, dim=1).cpu().numpy()
                    action_probs = torch.sigmoid(action_output).cpu().numpy()

                    ho_obj_probs = obj_prob[ho_infos[:, 2], :]
                    hoi_probs = np.zeros((ho_infos.shape[0], self.dataset.num_interactions))
                    for iid, (pid, oid) in enumerate(self.dataset.interactions):
                        hoi_probs[:, iid] = ho_obj_probs[:, oid] * action_probs[:, pid]

                    prediction.ho_img_inds = ho_infos[:, 0]
                    prediction.ho_pairs = ho_infos[:, 1:]
                    prediction.obj_prob = obj_prob
                    prediction.action_probs = action_probs
                    prediction.hoi_scores = hoi_probs

                if vis_output.boxes_ext is not None:
                    boxes_ext = vis_output.boxes_ext.cpu().numpy()
                    im_scales = x.img_infos[:, 2].cpu().numpy()

                    obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
                    obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
                    prediction.obj_im_inds = obj_im_inds
                    prediction.obj_boxes = obj_boxes

                return prediction

    def _forward(self, vis_output: VisualOutput):
        if vis_output.box_labels is not None:
            vis_output.filter_boxes(thr=None)
        boxes_ext = vis_output.boxes_ext
        box_feats = vis_output.box_feats
        masks = vis_output.masks
        union_boxes_feats = vis_output.hoi_union_boxes_feats
        hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)

        obj_repr = self.obj_branch(boxes_ext, box_feats)
        obj_logits = self.obj_output_fc(obj_repr)

        act_repr = self.act_branch(obj_repr, union_boxes_feats, hoi_infos)
        action_logits = self.action_output_fc(act_repr)

        return obj_logits, action_logits


class KatoModel(GenericModel):
    # FIXME?
    @classmethod
    def get_cline_name(cls):
        return 'kato'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        raise NotImplementedError('This needs to be checked after refactors.')
        self.obj_branch = SimpleObjBranch(input_dim=self.visual_module.vis_feat_dim + self.dataset.num_object_classes)
        self.obj_output_fc = nn.Linear(self.obj_branch.output_dim, self.dataset.num_object_classes)

        self.hoi_branch = KatoGCNBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_dim, dataset)

    def _forward(self):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_repr = self.obj_branch(boxes_ext, box_feats)
        obj_logits = self.obj_output_fc(obj_repr)

        hoi_obj_logits, action_logits = self.hoi_branch(obj_repr, union_boxes_feats, hoi_infos)

        return obj_logits, action_logits


class PeyreModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'peyre'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        raise NotImplementedError('This needs to be checked after refactors.')
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
                logits = self._forward()
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

    def _forward(self):
        hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits = self.hoi_branch(boxes_ext, box_feats, hoi_infos)
        return hoi_subj_logits, hoi_obj_logits, hoi_act_logits, hoi_logits

    def _prepare_prediction(self, obj_output, action_output, hoi_probs, hoi_infos, boxes_ext, im_scales):
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
