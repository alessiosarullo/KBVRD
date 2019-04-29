from lib.models.generic_model import GenericModel, Prediction
from lib.dataset.utils import Splits
from lib.bbox_utils import compute_ious
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

    def forward(self, x, inference=True, **kwargs):
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


class ActionOnlyModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'actonly'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = SimpleObjBranch(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.hoi_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.repr_dim)

        self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)
        self.action_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.action_output_fc.weight, gain=1.0)

        self.hoi_refinement_branch = HoiPriorBranch(dataset, vis_feat_dim)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids)
        obj_logits = self.obj_output_fc(obj_repr)

        hoi_repr = self.hoi_branch(obj_repr, union_boxes_feats, hoi_infos)
        action_logits = self.action_output_fc(hoi_repr)

        return obj_logits, action_logits, None


class HoiOnlyModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'hoionly'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = SimpleObjBranch(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.hoi_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.repr_dim)

        self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_interactions, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        self.hoi_refinement_branch = HoiPriorBranch(dataset, vis_feat_dim)

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


class EmbsimModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'embsimact'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.obj_branch = SimpleObjBranch(input_dim=self.visual_module.vis_feat_dim + self.dataset.num_object_classes)
        self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)
        # torch.nn.init.xavier_normal_(self.obj_output_fc.weight, gain=1.0)

        self.hoi_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.repr_dim)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        self.hoi_embsim_branch = HoiEmbsimBranch(self.visual_module.vis_feat_dim, dataset)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids)
        obj_logits = self.obj_output_fc(obj_repr)
        hoi_repr = self.hoi_branch(boxes_ext, obj_repr, union_boxes_feats, hoi_infos, obj_logits, box_labels)
        action_logits = self.hoi_output_fc(hoi_repr)

        action_logits_emb, hoi_obj_logits = self.hoi_embsim_branch(union_boxes_feats, box_feats, hoi_infos)

        obj_to_hoi_matrix = hoi_obj_logits.new_zeros(boxes_ext.shape[0], hoi_infos.shape[0])
        obj_to_hoi_matrix[hoi_infos[:, 2], torch.arange(hoi_infos.shape[0])] = 1
        obj_logits_hoi = obj_to_hoi_matrix @ hoi_obj_logits / (obj_to_hoi_matrix.sum(dim=1, keepdim=True).clamp(min=1))
        obj_logits = obj_logits + obj_logits_hoi

        action_logits = action_logits + action_logits_emb

        return obj_logits, action_logits


class KatoModel(GenericModel):
    # FIXME?
    @classmethod
    def get_cline_name(cls):
        return 'kato'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.obj_branch = SimpleObjBranch(input_dim=self.visual_module.vis_feat_dim + self.dataset.num_object_classes)
        self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)

        self.hoi_branch = KatoGCNBranch(self.visual_module.vis_feat_dim, self.obj_branch.repr_dim, dataset)

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
    # FIXME
    @classmethod
    def get_cline_name(cls):
        return 'peyre'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.hoi_branch = PeyreEmbsimBranch(self.visual_module.vis_feat_dim, dataset)

    def get_losses(self, x, **kwargs):
        obj_output, hoi_output, box_labels, action_labels = self(x, inference=False, **kwargs)
        box_labels_1hot = box_labels.new_zeros((box_labels.shape[0], self.dataset.num_object_classes)).float()
        box_labels_1hot[torch.arange(box_labels_1hot.shape[0]), box_labels] = 1
        obj_loss = nn.functional.binary_cross_entropy_with_logits(obj_output, box_labels_1hot) * self.dataset.num_object_classes
        hoi_loss = nn.functional.binary_cross_entropy_with_logits(hoi_output, action_labels) * self.dataset.num_predicates
        return {'object_loss': obj_loss, 'hoi_loss': hoi_loss}

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, action_labels=None, hoi_labels=None):
        # hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        obj_logits, action_logits = self.hoi_branch(boxes_ext, box_feats, hoi_infos)
        return obj_logits, action_logits


def main():
    from lib.dataset.hicodet import Splits
    from scripts.utils import get_all_models_by_name

    cfg.parse_args(allow_required=False)
    hdtrain = HicoDetInstanceSplit.get_split(split=Splits.TRAIN)
    detector = get_all_models_by_name()[cfg.program.model](hdtrain)
    detector.cuda()


if __name__ == '__main__':
    main()
