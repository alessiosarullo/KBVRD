import torch
import torch.nn as nn

from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.models.context_modules import SpatialContext, ObjectContext
from lib.models.generic_model import GenericModel
from lib.models.hoi_branches import KBNMotifsHOIBranch, MemHoiBranch, HoiPriorBranch


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


class HoiModel(GenericModel):
    @classmethod
    def get_cline_name(cls):
        return 'hoi'

    def __init__(self, dataset: HicoDetInstanceSplit, **kwargs):
        super().__init__(dataset, **kwargs)
        vis_feat_dim = self.visual_module.vis_feat_dim
        self.obj_branch = ObjectContext(input_dim=vis_feat_dim + self.dataset.num_object_classes)

        self.post_lstm = nn.Linear(self.obj_branch.repr_dim, vis_feat_dim)
        torch.nn.init.xavier_normal_(self.post_lstm.weight, gain=1.0)

        self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(vis_feat_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        self.hoi_refinement_branch = HoiPriorBranch(dataset, vis_feat_dim)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_ctx, obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids, spatial_ctx=None)

        obj_logits = self.obj_output_fc(obj_repr)

        hoi_repr = self.post_lstm(obj_repr[hoi_infos[:, 2], :]) + union_boxes_feats
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
        self.spatial_context_branch = SpatialContext(input_dim=2 * (self.visual_module.mask_resolution ** 2))
        self.obj_branch = ObjectContext(input_dim=self.visual_module.vis_feat_dim +
                                                  self.dataset.num_object_classes +
                                                  self.spatial_context_branch.context_dim)
        self.hoi_branch = KBNMotifsHOIBranch(self.visual_module.vis_feat_dim, self.obj_branch.repr_dim, dataset)

        self.obj_output_fc = nn.Linear(self.obj_branch.repr_dim, self.dataset.num_object_classes)
        self.hoi_output_fc = nn.Linear(self.hoi_branch.output_dim, dataset.num_predicates, bias=True)
        torch.nn.init.xavier_normal_(self.hoi_output_fc.weight, gain=1.0)

        self.hoi_refinement_branch = HoiPriorBranch(dataset, self.hoi_branch.output_dim)

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        spatial_ctx, spatial_repr = self.spatial_context_branch(masks, im_ids, hoi_infos)
        obj_ctx, obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids, spatial_ctx)

        obj_logits = self.obj_output_fc(obj_repr)
        hoi_repr = self.hoi_branch(boxes_ext, obj_repr, union_boxes_feats, hoi_infos, box_labels)
        hoi_logits = self.hoi_output_fc(hoi_repr)
        hoi_logits = self.hoi_refinement_branch(hoi_logits, hoi_repr, boxes_ext, hoi_infos, box_labels)

        for k, v in self.hoi_refinement_branch.values_to_monitor.items():  # FIXME delete
            self.values_to_monitor[k] = v

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
        mem_margin_loss = (mem_margin + 0.1).clamp(min=0)  # FIXME magic constant
        return {'object_loss': obj_loss, 'hoi_loss': hoi_loss, 'mem_loss': mem_loss}

    def forward(self, x, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, hoi_infos, box_labels, hoi_labels = self.visual_module(x, inference)
            # `hoi_infos` is an R x 3 NumPy array where each column is [image ID, subject index, object index].
            # Masks are floats at this point.

            if hoi_infos is not None:
                obj_output, hoi_output, mem_output = self._forward(boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels, hoi_labels)
            else:
                obj_output = hoi_output = mem_output = None

            if not inference:
                assert obj_output is not None and hoi_output is not None and mem_output is not None \
                       and box_labels is not None and hoi_labels is not None
                return obj_output, hoi_output, mem_output, box_labels, hoi_labels
            else:
                assert mem_output is None
                return self._prepare_prediction(obj_output, hoi_output, hoi_infos, boxes_ext, im_scales=x.img_infos[:, 2].cpu().numpy())

    def _forward(self, boxes_ext, box_feats, masks, union_boxes_feats, hoi_infos, box_labels=None, hoi_labels=None):
        box_im_ids = boxes_ext[:, 0].long()
        hoi_infos = torch.tensor(hoi_infos, device=masks.device)
        im_ids = torch.unique(hoi_infos[:, 0], sorted=True)
        box_unique_im_ids = torch.unique(box_im_ids, sorted=True)
        assert im_ids.equal(box_unique_im_ids), (im_ids, box_unique_im_ids)

        obj_ctx, obj_repr = self.obj_branch(boxes_ext, box_feats, im_ids, box_im_ids, spatial_ctx=None)

        obj_logits = self.obj_output_fc(obj_repr)

        hoi_repr, mem_pred = self.hoi_branch(boxes_ext, obj_repr, union_boxes_feats, hoi_infos, box_labels, hoi_labels)
        hoi_logits = self.hoi_output_fc(hoi_repr)
        hoi_logits = self.hoi_refinement_branch(hoi_logits, hoi_repr, boxes_ext, hoi_infos, box_labels)

        if mem_pred is not None:
            mem_output = self.mem_output_fc(mem_pred)
        else:
            mem_output = None

        return obj_logits, hoi_logits, mem_output
