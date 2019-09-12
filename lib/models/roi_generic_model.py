import numpy as np
import torch

from torch import  nn

from config import cfg
from lib.dataset.hicodet.hicodet_singlehois_split import HicoDetSingleHOIsSplit
from lib.dataset.hicodet.hicodet_split import PrecomputedMinibatch
from lib.models.abstract_model import AbstractModel
from lib.models.containers import Prediction, VisualOutput
from lib.models.misc import bce_loss


class GenericModel(AbstractModel):
    @classmethod
    def get_cline_name(cls) -> str:
        raise NotImplementedError()

    def __init__(self, dataset: HicoDetSingleHOIsSplit, **kwargs):
        # Instance parameters defined before superclass' constructor's invocation will be updated according to keyword arguments.
        super().__init__(**kwargs)
        self.dataset = dataset  # type: HicoDetSingleHOIsSplit
        self.vis_feat_dim = self.dataset.precomputed_visual_feat_dim

        if cfg.csp:
            hist = np.sum(self.dataset.pc_action_labels, axis=0)
            if not cfg.train_null:
                hist[0] = 0
            cost_matrix = np.maximum(1, np.log2(hist[None, :] / hist[:, None])) * (1 - np.eye(hist.size))
            csp_weights = cost_matrix @ hist / (hist.sum() - hist)
            self.csp_weights = nn.Parameter(torch.from_numpy(csp_weights).float(), requires_grad=False)
        else:
            self.csp_weights = None

    @property
    def final_repr_dim(self):
        raise NotImplementedError()

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.build_visual_output(x, inference)  # type: VisualOutput

            if vis_output.ho_infos_np is not None:
                outputs = self._forward(vis_output, epoch=x.epoch, step=x.iter)
                outputs = self._refine_output(x, inference, vis_output, outputs)
            else:
                assert inference
                outputs = None

            if not inference:
                return self._get_losses(vis_output, outputs)
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
                        prediction.ho_img_inds = vis_output.ho_infos_np[:, 0]
                        prediction.ho_pairs = vis_output.ho_infos_np[:, 1:]

                        assert outputs is not None
                        self._finalize_prediction(prediction, vis_output, outputs)
                return prediction

    def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):
        raise NotImplementedError()

    def _refine_output(self, x: PrecomputedMinibatch, inference, vis_output, outputs):
        return outputs

    def _get_losses(self, vis_output: VisualOutput, outputs):
        output = outputs
        if cfg.phoi:
            losses = {'hoi_loss': bce_loss(output, vis_output.hoi_labels)}
        else:
            losses = {'action_loss': bce_loss(output, vis_output.action_labels)}
        return losses

    # noinspection PyUnresolvedReferences
    def _finalize_prediction(self, prediction: Prediction, vis_output: VisualOutput, outputs):
        output = outputs
        if cfg.phoi:
            ho_obj_scores = prediction.obj_scores[vis_output.ho_infos_np[:, 2], :]
            hoi_obj_scores = ho_obj_scores[:, self.dataset.full_dataset.interactions[:, 1]]  # This helps
            prediction.hoi_scores = torch.sigmoid(output).cpu().numpy() * hoi_obj_scores
        else:
            if output.shape[1] < self.dataset.full_dataset.num_actions:
                assert output.shape[1] == self.dataset.num_actions
                restricted_action_output = output
                output = restricted_action_output.new_zeros((output.shape[0], self.dataset.full_dataset.num_actions))
                output[:, self.dataset.active_actions] = restricted_action_output
            prediction.action_scores = torch.sigmoid(output).cpu().numpy()

    # noinspection PyCallingNonCallable
    def build_visual_output(self, batch: PrecomputedMinibatch, inference):
        output = VisualOutput()
        training = not inference

        boxes_ext_np = batch.pc_boxes_ext
        ho_infos = batch.pc_ho_infos

        if boxes_ext_np.shape[0] > 0 or training:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            output.boxes_ext = torch.tensor(boxes_ext_np, device=device)
            output.box_feats = torch.tensor(batch.pc_box_feats, device=device)
            output.hoi_union_boxes = batch.pc_ho_union_boxes
            output.hoi_union_boxes_feats = torch.tensor(batch.pc_ho_union_feats, device=device)

            if ho_infos.shape[0] > 0:
                ho_infos = ho_infos.astype(np.int, copy=False)
                output.ho_infos_np = ho_infos

                if batch.pc_box_labels is not None:
                    box_labels_np = batch.pc_box_labels
                    action_labels_np = batch.pc_action_labels
                    output.box_labels = torch.tensor(box_labels_np, device=device)
                    output.action_labels = torch.tensor(action_labels_np, device=device)
                    output.hoi_labels = torch.tensor(self.action_labels_to_hoi_labels(box_labels_np, action_labels_np, ho_infos), device=device)
            else:
                assert inference

        # Note that box indices in `ho_infos` are over all boxes, NOT relative to each specific image
        return output

    def action_labels_to_hoi_labels(self, box_labels, action_labels, ho_infos):
        # Requires everything Numpy
        interactions = self.dataset.full_dataset.interactions
        hoi_obj_labels = box_labels[ho_infos[:, 2]]

        obj_labels_1hot = np.zeros((hoi_obj_labels.shape[0], self.dataset.full_dataset.num_objects), dtype=np.float32)
        fg_objs = np.flatnonzero(hoi_obj_labels >= 0)
        obj_labels_1hot[fg_objs, hoi_obj_labels[fg_objs]] = 1

        hoi_labels = obj_labels_1hot[:, interactions[:, 1]] * action_labels[:, interactions[:, 0]]
        assert hoi_labels.shape[0] == action_labels.shape[0] and hoi_labels.shape[1] == self.dataset.full_dataset.num_interactions
        return hoi_labels

    # def get_geo_feats(self, vis_output: VisualOutput, batch: PrecomputedMinibatch):
    #     boxes_ext = vis_output.boxes_ext
    #     hoi_infos = vis_output.ho_infos
    #
    #     im_sizes = torch.tensor(np.array([d['im_size'][::-1] * d['im_scale'] for d in batch.other_ex_data]).astype(np.float32),
    #                             device=boxes_ext.device)
    #     im_areas = im_sizes.prod(dim=1)
    #
    #     box_im_inds = boxes_ext[:, 0].long()
    #     box_im_sizes = im_sizes[box_im_inds, :]
    #
    #     norm_boxes = boxes_ext[:, 1:5] / box_im_sizes.repeat(1, 2)
    #     assert (0 <= norm_boxes).all(), \
    #         (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(),
    #          im_sizes.detach().cpu().numpy(), norm_boxes.detach().cpu().numpy())
    #     # norm_boxes.clamp_(max=1)  # Needed for numerical errors
    #     assert (norm_boxes <= 1).all(), \
    #         (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(),
    #          im_sizes.detach().cpu().numpy(), norm_boxes.detach().cpu().numpy())
    #
    #     box_widths = boxes_ext[:, 3] - boxes_ext[:, 1]
    #     box_heights = boxes_ext[:, 4] - boxes_ext[:, 2]
    #     norm_box_areas = box_widths * box_heights / im_areas[box_im_inds]
    #     assert (0 < norm_box_areas).all(), \
    #         (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(), norm_box_areas.detach().cpu().numpy())
    #     # norm_box_areas.clamp_(max=1)  # Needed for numerical errors
    #     assert (norm_box_areas <= 1).all(), \
    #         (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(), norm_box_areas.detach().cpu().numpy())
    #
    #     hum_inds = hoi_infos[:, 1]
    #     obj_inds = hoi_infos[:, 2]
    #     obj_widths = box_widths[obj_inds]
    #     obj_heights = box_widths[obj_inds]
    #
    #     h_dist = (boxes_ext[hum_inds, 1] - boxes_ext[obj_inds, 1]) / obj_widths
    #     v_dist = (boxes_ext[hum_inds, 2] - boxes_ext[obj_inds, 2]) / obj_heights
    #
    #     h_ratio = (box_widths[hum_inds] / obj_widths).log()
    #     v_ratio = (box_widths[hum_inds] / obj_heights).log()
    #
    #     geo_feats = torch.cat([norm_boxes[hum_inds, :],
    #                            norm_box_areas[hum_inds, None],
    #                            norm_boxes[obj_inds, :],
    #                            norm_box_areas[obj_inds, None],
    #                            h_dist[:, None], v_dist[:, None], h_ratio[:, None], v_ratio[:, None]
    #                            ], dim=1)
    #     return geo_feats
