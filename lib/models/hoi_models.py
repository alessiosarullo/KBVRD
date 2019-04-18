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

        boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, hoi_infos, box_labels, hoi_labels = self.visual_module(x, True)
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

            hoi_labels = np.zeros((predict_ho_pairs.shape[0], self.dataset.num_predicates))
            for predict_idx, (ph, po) in enumerate(predict_ho_pairs):
                gt_pair_ious = np.zeros(gt_hois.shape[0])
                for gtidx, (gh, go, gi) in enumerate(gt_hois):
                    iou_h = pred_gt_ious[ph, gh]
                    iou_o = pred_gt_ious[po, go]
                    if pred_gt_ious_class_match[ph, gh] and pred_gt_ious_class_match[po, go]:
                        gt_pair_ious[gtidx] = min(iou_h, iou_o)
                if np.any(gt_pair_ious > self.iou_thresh):
                    gtidxs = (gt_pair_ious > self.iou_thresh)
                    hoi_labels[predict_idx, np.unique(gt_hois[gtidxs, 2])] = 1

            hoi_output = torch.from_numpy(hoi_labels)
        else:
            obj_output = hoi_output = boxes_ext = None

        if not inference:
            assert obj_output is not None and hoi_output is not None and box_labels is not None and hoi_labels is not None
            return obj_output, hoi_output, box_labels, hoi_labels
        else:
            return self._prepare_prediction(obj_output, hoi_output, hoi_infos, boxes_ext, im_scales=im_scales)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        raise NotImplementedError()

    @staticmethod
    def _prepare_prediction(obj_output, hoi_output, hoi_infos, boxes_ext, im_scales):
        if hoi_infos is not None:
            assert obj_output is not None and hoi_output is not None and boxes_ext is not None
            obj_prob = obj_output.cpu().numpy()
            hoi_probs = hoi_output.cpu().numpy()
            hoi_img_inds = hoi_infos[:, 0]
            ho_pairs = hoi_infos[:, 1:]
        else:
            hoi_probs = ho_pairs = hoi_img_inds = None
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
                          hoi_img_inds=hoi_img_inds,
                          ho_pairs=ho_pairs,
                          hoi_scores=hoi_probs)


class PureMemModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'puremem'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        feats = np.empty((dataset.precomputed_visual_feat_dim, dataset.num_precomputed_hois), dtype=np.float32)
        labels = np.empty((dataset.num_precomputed_hois, dataset.num_predicates), dtype=np.float32)
        idx = 0
        for i in range(dataset.num_images):
            ex = dataset.get_entry(i)
            ex_feats, ex_labels = ex.precomp_hoi_union_feats, ex.precomp_hoi_labels
            assert ex_feats.shape[0] == ex_labels.shape[0]
            n = ex_feats.shape[0]
            feats[:, idx:idx + n] = (ex_feats / np.linalg.norm(ex_feats, axis=1, keepdims=True)).T
            labels[idx:idx + n, :] = ex_labels
            idx += n
        self.feats = torch.nn.Parameter(torch.from_numpy(feats), requires_grad=False)
        self.labels = torch.nn.Parameter(torch.from_numpy(labels), requires_grad=False)

    def get_losses(self, x, **kwargs):
        raise NotImplementedError()

    def forward(self, x, inference=True, **kwargs):
        assert inference is True
        assert not self.training

        boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, hoi_infos, box_labels, hoi_labels = self.visual_module(x, inference)

        if hoi_infos is not None:
            hoi_output = self._forward(None, None, None, union_boxes_feats, None)
        else:
            hoi_output = None

        return self._prepare_prediction(None, hoi_output, hoi_infos, boxes_ext, im_scales=x.img_infos[:, 2].cpu().numpy())

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        union_boxes_feats_norm = torch.nn.functional.normalize(union_boxes_feats)
        sim = torch.nn.functional.normalize(union_boxes_feats_norm @ self.feats)
        hoi_output = sim @ self.labels
        return hoi_output

    @staticmethod
    def _prepare_prediction(obj_output, hoi_output, hoi_infos, boxes_ext, im_scales):
        assert obj_output is None
        if hoi_infos is not None:
            assert hoi_output is not None and boxes_ext is not None
            obj_prob = None  # this will be assigned later as the object label distribution
            hoi_probs = hoi_output.cpu().numpy()
            hoi_img_inds = hoi_infos[:, 0]
            ho_pairs = hoi_infos[:, 1:]
        else:
            hoi_probs = ho_pairs = hoi_img_inds = None
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
                          hoi_img_inds=hoi_img_inds,
                          ho_pairs=ho_pairs,
                          hoi_scores=hoi_probs)


class ZeroModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'zero'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.obj_output_fc = nn.Linear(self.visual_module.vis_feat_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(self.visual_module.vis_feat_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_logits = self.obj_output_fc(box_feats)
        hoi_logits = self.hoi_output_fc(box_feats[hoi_infos[:, 2], :])

        return obj_logits, hoi_logits


class HoiModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'hoi'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = SimpleObjBranch(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.hoi_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.repr_dim)

        self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        self.hoi_refinement_branch = HoiPriorBranch(dataset, vis_feat_dim)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids)
        obj_logits = self.obj_output_fc(obj_repr)

        hoi_repr = self.hoi_branch(boxes_ext, obj_repr, union_boxes_feats, hoi_infos, obj_logits, box_labels)
        hoi_logits = self.hoi_output_fc(hoi_repr)

        hoi_logits = self.hoi_refinement_branch(hoi_logits, hoi_repr, boxes_ext, hoi_infos, box_labels)
        for k, v in self.hoi_refinement_branch.values_to_monitor.items():  # FIXME delete
            self.values_to_monitor[k] = v

        return obj_logits, hoi_logits


class EmbsimModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'embsim'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.obj_branch = SimpleObjBranch(input_dim=self.visual_module.vis_feat_dim + self.dataset.num_object_classes)
        self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)
        # torch.nn.init.xavier_normal_(self.obj_output_fc.weight, gain=1.0)

        self.hoi_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.repr_dim)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        self.hoi_refinement_branch = HoiEmbsimBranch(self.visual_module.vis_feat_dim, dataset)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids)
        obj_logits = self.obj_output_fc(obj_repr)
        hoi_repr = self.hoi_branch(boxes_ext, obj_repr, union_boxes_feats, hoi_infos, obj_logits, box_labels)
        hoi_logits = self.hoi_output_fc(hoi_repr)

        hoi_logits2, hoi_obj_logits = self.hoi_refinement_branch(union_boxes_feats, box_feats, hoi_infos)

        obj_to_hoi_matrix = hoi_obj_logits.new_zeros(boxes_ext.shape[0], hoi_infos.shape[0])
        obj_to_hoi_matrix[hoi_infos[:, 2], torch.arange(hoi_infos.shape[0])] = 1
        obj_logits_hoi = obj_to_hoi_matrix @ hoi_obj_logits / (obj_to_hoi_matrix.sum(dim=1, keepdim=True).clamp(min=1))
        obj_logits = obj_logits + obj_logits_hoi

        hoi_logits = hoi_logits + hoi_logits2

        return obj_logits, hoi_logits


class PeyreModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'peyre'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.hoi_branch = PeyreEmbsimBranch(self.visual_module.vis_feat_dim, dataset)

    def get_losses(self, x, **kwargs):
        obj_output, hoi_output, box_labels, hoi_labels = self(x, inference=False, **kwargs)
        obj_loss = nn.functional.binary_cross_entropy_with_logits(obj_output, box_labels) * self.dataset.num_object_classes
        hoi_loss = nn.functional.binary_cross_entropy_with_logits(hoi_output, hoi_labels) * self.dataset.num_predicates
        return {'object_loss': obj_loss, 'hoi_loss': hoi_loss}

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        # hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        obj_logits, hoi_logits = self.hoi_branch(boxes_ext, box_feats, hoi_infos)
        return obj_logits, hoi_logits


def main():
    from lib.dataset.hicodet import Splits
    from scripts.utils import get_all_models_by_name

    cfg.parse_args(allow_required=False)
    hdtrain = HicoDetInstanceSplit.get_split(split=Splits.TRAIN)
    detector = get_all_models_by_name()[cfg.program.model](hdtrain)
    detector.cuda()


if __name__ == '__main__':
    main()
