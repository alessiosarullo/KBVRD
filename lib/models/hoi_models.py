from lib.models.generic_model import GenericModel, Prediction
from lib.models.hoi_branches import *
from lib.models.obj_branches import *


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
            feats[:, idx:idx + n] = ex_feats.T
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
        sim = union_boxes_feats @ self.feats
        hoi_output = sim @ self.labels
        return hoi_output

    @staticmethod
    def _prepare_prediction(obj_output, hoi_output, hoi_infos, boxes_ext, im_scales):
        assert obj_output is None
        if hoi_infos is not None:
            assert hoi_output is not None and boxes_ext is not None
            obj_prob = None  # this will be assigned later as the object label distribution
            hoi_probs = hoi_output
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


class ObjModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'obj'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = ObjectContext(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(self.obj_branch.repr_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_ctx, obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids, spatial_ctx=None)

        obj_logits = self.obj_output_fc(obj_repr)
        hoi_logits = self.hoi_output_fc(obj_repr[hoi_infos[:, 2], :])

        return obj_logits, hoi_logits


class NoObjModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'noobj'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.hoi_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, vis_feat_dim)

        self.obj_output_fc = nn.Linear(vis_feat_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_logits = self.obj_output_fc(box_feats)

        hoi_repr = self.hoi_branch(boxes_ext, box_feats, union_boxes_feats, hoi_infos, obj_logits, box_labels)
        hoi_logits = self.hoi_output_fc(hoi_repr)

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


class KBModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'kb'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        self.obj_branch = ObjectContext(input_dim=self.visual_module.vis_feat_dim + self.dataset.num_object_classes)

        self.hoi_branch = KBHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.repr_dim, dataset)

        self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(self.visual_module.vis_feat_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        self.hoi_refinement_branch = HoiPriorBranch(dataset, self.visual_module.vis_feat_dim)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_ctx, obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids, spatial_ctx=None)

        obj_logits = self.obj_output_fc(obj_repr)

        hoi_repr = self.hoi_branch(boxes_ext, obj_repr, union_boxes_feats, hoi_infos, obj_logits, box_labels)
        hoi_logits = self.hoi_output_fc(hoi_repr)

        hoi_logits = self.hoi_refinement_branch(hoi_logits, hoi_repr, boxes_ext, hoi_infos, box_labels)
        for k, v in self.hoi_refinement_branch.values_to_monitor.items():  # FIXME delete
            self.values_to_monitor[k] = v

        return obj_logits, hoi_logits


class KVMemoryModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'kvm'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = ObjectContext(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.hoi_branch = Mem2HoiBranch(vis_feat_dim, self.obj_branch.repr_dim, dataset)

        self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        self.mem_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        self.hoi_refinement_branch = HoiPriorBranch(dataset, self.hoi_branch.output_dim)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_ctx, obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids, spatial_ctx=None)

        obj_logits = self.obj_output_fc(obj_repr)

        hoi_repr = self.hoi_branch(boxes_ext, obj_repr, union_boxes_feats, hoi_infos, box_labels, hoi_labels)
        hoi_logits = self.hoi_output_fc(hoi_repr)
        hoi_logits = self.hoi_refinement_branch(hoi_logits, hoi_repr, boxes_ext, hoi_infos, box_labels)

        return obj_logits, hoi_logits


class MemoryModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'mem'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = ObjectContext(input_dim=vis_feat_dim + self.dataset.num_object_classes)
        self.hoi_branch = MemHoiBranch(vis_feat_dim, self.obj_branch.repr_dim, dataset)

        self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        self.mem_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        self.hoi_refinement_branch = HoiPriorBranch(dataset, self.hoi_branch.output_dim)

    def get_losses(self, x, **kwargs):
        obj_output, hoi_output, mem_outputs, box_labels, hoi_labels = self(x, inference=False, **kwargs)
        obj_loss = nn.functional.cross_entropy(obj_output, box_labels)
        hoi_loss = nn.functional.binary_cross_entropy_with_logits(hoi_output, hoi_labels) * self.dataset.num_predicates

        mem_output, mem_margin = mem_outputs
        mem_loss = nn.functional.binary_cross_entropy_with_logits(mem_output, hoi_labels) * self.dataset.num_predicates
        mem_margin_loss = (mem_margin + 0.6).clamp(min=0).mean()  # FIXME magic constant
        return {'object_loss': obj_loss, 'hoi_loss': hoi_loss, 'mem_loss': mem_loss, 'mem_marg_loss': mem_margin_loss}

    def forward(self, x, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, hoi_infos, box_labels, hoi_labels = self.visual_module(x, inference)
            # `hoi_infos` is an R x 3 NumPy array where each column is [image ID, subject index, object index].
            # Masks are floats at this point.

            if hoi_infos is not None:
                obj_output, hoi_output, mem_outputs = self._forward(boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels, hoi_labels)
            else:
                obj_output = hoi_output = mem_outputs = None

            if not inference:
                assert obj_output is not None and hoi_output is not None and mem_outputs is not None \
                       and box_labels is not None and hoi_labels is not None
                return obj_output, hoi_output, mem_outputs, box_labels, hoi_labels
            else:
                assert mem_outputs is None
                return self._prepare_prediction(obj_output, hoi_output, hoi_infos, boxes_ext, im_scales=x.img_infos[:, 2].cpu().numpy())

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_ctx, obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids, spatial_ctx=None)

        obj_logits = self.obj_output_fc(obj_repr)

        hoi_repr, mem_pred, margin_loss = self.hoi_branch(boxes_ext, obj_repr, union_boxes_feats, hoi_infos, box_labels, hoi_labels)
        hoi_logits = self.hoi_output_fc(hoi_repr)
        hoi_logits = self.hoi_refinement_branch(hoi_logits, hoi_repr, boxes_ext, hoi_infos, box_labels)

        if mem_pred is not None:
            mem_pred = self.mem_output_fc(mem_pred)
            mem_outputs = (mem_pred, margin_loss)
        else:
            assert mem_pred is None and margin_loss is None
            mem_outputs = None

        return obj_logits, hoi_logits, mem_outputs


def main():
    from lib.dataset.hicodet import Splits
    from scripts.utils import get_all_models_by_name

    cfg.parse_args(allow_required=False)
    hdtrain = HicoDetInstanceSplit.get_split(split=Splits.TRAIN)
    detector = get_all_models_by_name()[cfg.program.model](hdtrain)
    detector.cuda()


if __name__ == '__main__':
    main()
