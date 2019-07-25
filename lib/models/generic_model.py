import numpy as np
import torch

from config import cfg
from lib.dataset.hicodet.hicodet_split import HicoDetSplit
from lib.dataset.hicodet.pc_hicodet_split import PrecomputedMinibatch
from lib.detection.visual_module import VisualModule
from lib.models.abstract_model import AbstractModel
from lib.models.containers import Prediction, VisualOutput
from lib.models.misc import bce_loss


class GenericModel(AbstractModel):
    @classmethod
    def get_cline_name(cls) -> str:
        raise NotImplementedError()

    def __init__(self, dataset: HicoDetSplit, **kwargs):
        # Instance parameters defined before superclass' constructor's invocation will be updated according to keyword arguments.
        super().__init__(**kwargs)
        self.dataset = dataset
        self.visual_module = VisualModule(dataset)

        # if cfg.csloss:
        #     prcls_hist = Counter(dataset.hoi_triplets[:, 1])
        #     prcls_hist = np.array([prcls_hist[i] for i in range(dataset.num_predicates)])
        #     num_predicate_classes = prcls_hist.size
        #     assert num_predicate_classes == dataset.num_predicates
        #     cost_matrix = np.maximum(1, np.log2(prcls_hist[None, :] / prcls_hist[:, None]))
        #     assert not np.any(np.isnan(cost_matrix))
        #     cost_matrix[np.arange(num_predicate_classes), np.arange(num_predicate_classes)] = 0
        #
        #     tot_num_preds = sum(prcls_hist)
        #     tot_other_preds = tot_num_preds - prcls_hist
        #     expected_class_cost = (cost_matrix.dot(prcls_hist)) / tot_other_preds
        #
        #     self.class_pos_weights = torch.nn.Parameter(torch.from_numpy(expected_class_cost).view(1, -1), requires_grad=False)
        #     self.class_neg_weights = torch.nn.Parameter(torch.from_numpy(cost_matrix), requires_grad=False)

    @property
    def final_repr_dim(self):
        raise NotImplementedError()

    #
    # def weighted_binary_cross_entropy_with_logits(self, logits, labels, num_rels=None):
    #     if num_rels is None:
    #         num_rels = self.dataset.num_predicates
    #     if not (labels.size() == logits.size()):
    #         raise ValueError("Target size ({}) must be the same as input size ({})".format(labels.size(), logits.size()))
    #
    #     # Binary cross entropy with the addition of two sets of class weights. One set is used for positive examples the other one is used for
    #     # negative examples.
    #     m = logits.clamp(min=0)
    #     s = logits
    #     t = labels
    #     u = self.class_pos_weights
    #     v = self.class_neg_weights[labels, :]  # FIXME
    #
    #     le = ((-m).exp() + (s - m).exp()).log()
    #
    #     # Trust
    #     if u is not None and v is not None:
    #         loss = v * m - t * (v * m + u * (s - m)) + ((1 - t) * v + u * t) * le
    #     elif u is not None and v is None:
    #         loss = m - t * (m + u * (s - m)) + (1 - t + u * t) * le
    #     elif u is None and v is not None:
    #         loss = v * m - t * (v * m + s - m) + ((1 - t) * v + t) * le
    #     else:
    #         loss = F.binary_cross_entropy_with_logits(logits, labels)
    #
    #     loss = loss.mean() * num_rels  # The average is computed over classes and examples, instead of only over examples. This fixes it.
    #     return loss

    def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_output = self.visual_module(x, inference)  # type: VisualOutput

            if vis_output.ho_infos_np is not None:
                outputs = self._forward(vis_output, batch=x, epoch=x.epoch, step=x.iter)
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

    def _finalize_prediction(self, prediction: Prediction, vis_output: VisualOutput, outputs):
        output = outputs
        if cfg.phoi:
            ho_obj_scores = prediction.obj_scores[vis_output.ho_infos_np[:, 2], :]
            hoi_obj_scores = ho_obj_scores[:, self.dataset.hicodet.interactions[:, 1]]  # This helps
            prediction.hoi_scores = torch.sigmoid(output).cpu().numpy() * hoi_obj_scores
        else:
            if output.shape[1] < self.dataset.hicodet.num_predicates:
                assert output.shape[1] == self.dataset.num_predicates
                restricted_action_output = output
                output = restricted_action_output.new_zeros((output.shape[0], self.dataset.hicodet.num_predicates))
                output[:, self.dataset.active_predicates] = restricted_action_output
            prediction.action_scores = torch.sigmoid(output).cpu().numpy()

    def get_geo_feats(self, vis_output: VisualOutput, batch: PrecomputedMinibatch):
        boxes_ext = vis_output.boxes_ext
        hoi_infos = vis_output.ho_infos

        im_sizes = torch.tensor(np.array([d['im_size'][::-1] * d['im_scale'] for d in batch.other_ex_data]).astype(np.float32),
                                device=boxes_ext.device)
        im_areas = im_sizes.prod(dim=1)

        box_im_inds = boxes_ext[:, 0].long()
        box_im_sizes = im_sizes[box_im_inds, :]

        norm_boxes = boxes_ext[:, 1:5] / box_im_sizes.repeat(1, 2)
        assert (0 <= norm_boxes).all(), \
            (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(),
             im_sizes.detach().cpu().numpy(), norm_boxes.detach().cpu().numpy())
        # norm_boxes.clamp_(max=1)  # Needed for numerical errors
        assert (norm_boxes <= 1).all(), \
            (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(),
             im_sizes.detach().cpu().numpy(), norm_boxes.detach().cpu().numpy())

        box_widths = boxes_ext[:, 3] - boxes_ext[:, 1]
        box_heights = boxes_ext[:, 4] - boxes_ext[:, 2]
        norm_box_areas = box_widths * box_heights / im_areas[box_im_inds]
        assert (0 < norm_box_areas).all(), \
            (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(), norm_box_areas.detach().cpu().numpy())
        # norm_box_areas.clamp_(max=1)  # Needed for numerical errors
        assert (norm_box_areas <= 1).all(), \
            (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(), norm_box_areas.detach().cpu().numpy())

        hum_inds = hoi_infos[:, 1]
        obj_inds = hoi_infos[:, 2]
        obj_widths = box_widths[obj_inds]
        obj_heights = box_widths[obj_inds]

        h_dist = (boxes_ext[hum_inds, 1] - boxes_ext[obj_inds, 1]) / obj_widths
        v_dist = (boxes_ext[hum_inds, 2] - boxes_ext[obj_inds, 2]) / obj_heights

        h_ratio = (box_widths[hum_inds] / obj_widths).log()
        v_ratio = (box_widths[hum_inds] / obj_heights).log()

        geo_feats = torch.cat([norm_boxes[hum_inds, :],
                               norm_box_areas[hum_inds, None],
                               norm_boxes[obj_inds, :],
                               norm_box_areas[obj_inds, None],
                               h_dist[:, None], v_dist[:, None], h_ratio[:, None], v_ratio[:, None]
                               ], dim=1)
        return geo_feats
