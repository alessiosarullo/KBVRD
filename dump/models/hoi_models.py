# from lib.bbox_utils import compute_ious
# from lib.dataset.utils import Splits
# from lib.models.generic_model import GenericModel, Prediction
# from lib.dataset.hicodet.pc_hicodet_split import PrecomputedMinibatch
# from lib.models.containers import VisualOutput
# from lib.models.hoi_branches import *
# from lib.dump.obj_branches import *
#
#
# class OracleModel(GenericModel):
#     @classmethod
#     def get_cline_name(cls):
#         return 'oracle'
#
#     def __init__(self, dataset: HicoDetSplit, **kwargs):
#         super().__init__(dataset, **kwargs)
#         raise NotImplementedError('This needs to be checked after refactors.')
#         self.fake = torch.nn.Parameter(torch.from_numpy(np.array([1.])), requires_grad=True)
#         self.iou_thresh = 0.5
#         self.split = Splits.TEST
#         self.perfect_detector = True
#
#     def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
#         assert inference is True
#         assert not self.training
#
#         boxes_ext, box_feats, masks, union_boxes, union_boxes_feats, hoi_infos, box_labels, action_labels, hoi_labels = self.visual_module(x, True)
#         im_scales = x.img_infos[:, 2].cpu().numpy()
#         gt_entry = HicoDetSplitBuilder.get_split(HicoDetSplit, self.split).get_img_entry(x.other_ex_data[0]['index'], read_img=False)
#         gt_boxes = gt_entry.gt_boxes * im_scales[0]
#         gt_obj_classes = gt_entry.gt_obj_classes
#         if self.perfect_detector:
#             boxes_ext = torch.from_numpy(np.concatenate([np.zeros((gt_boxes.shape[0], 1)),
#                                                          gt_boxes,
#                                                          self.visual_module.one_hot_obj_labels(gt_obj_classes)
#                                                          ], axis=1))
#             hoi_infos = self.visual_module.get_all_pairs(boxes_ext[:, :5].detach().cpu().numpy(), gt_obj_classes)
#             if hoi_infos.size == 0:
#                 hoi_infos = None
#
#         if hoi_infos is not None:
#             im_ids = np.unique(hoi_infos[:, 0])
#             assert im_ids.size == 1 and im_ids == 0
#
#             predict_boxes = boxes_ext[:, 1:5].detach().cpu().numpy()
#             pred_gt_ious = compute_ious(predict_boxes, gt_boxes)
#
#             pred_gt_best_match = np.argmax(pred_gt_ious, axis=1)  # type: np.ndarray
#             box_labels = gt_obj_classes[pred_gt_best_match]  # assign the best match
#             obj_output = torch.from_numpy(self.visual_module.one_hot_obj_labels(box_labels))
#
#             pred_gt_ious_class_match = (box_labels[:, None] == gt_obj_classes[None, :])
#
#             predict_ho_pairs = hoi_infos[:, 1:]
#             gt_hois = gt_entry.gt_hois[:, [0, 2, 1]]
#
#             action_labels = np.zeros((predict_ho_pairs.shape[0], self.dataset.num_predicates))
#             for predict_idx, (ph, po) in enumerate(predict_ho_pairs):
#                 gt_pair_ious = np.zeros(gt_hois.shape[0])
#                 for gtidx, (gh, go, gi) in enumerate(gt_hois):
#                     iou_h = pred_gt_ious[ph, gh]
#                     iou_o = pred_gt_ious[po, go]
#                     if pred_gt_ious_class_match[ph, gh] and pred_gt_ious_class_match[po, go]:
#                         gt_pair_ious[gtidx] = min(iou_h, iou_o)
#                 if np.any(gt_pair_ious > self.iou_thresh):
#                     gtidxs = (gt_pair_ious > self.iou_thresh)
#                     action_labels[predict_idx, np.unique(gt_hois[gtidxs, 2])] = 1
#
#             action_output = torch.from_numpy(action_labels)
#         else:
#             obj_output = action_output = boxes_ext = None
#
#         if not inference:
#             assert obj_output is not None and action_output is not None and box_labels is not None and action_labels is not None
#             return obj_output, action_output, box_labels, action_labels
#         else:
#             return self._prepare_prediction(obj_output, action_output, hoi_infos, boxes_ext, im_scales=im_scales)
#
#     def _forward(self, **kwargs):
#         raise NotImplementedError()
#
#     def _prepare_prediction(self, obj_output, action_output, hoi_output, hoi_infos, boxes_ext, im_scales):
#         if hoi_infos is not None:
#             assert obj_output is not None and action_output is not None and boxes_ext is not None
#             obj_prob = obj_output.cpu().numpy()
#             action_probs = action_output.cpu().numpy()
#             ho_img_inds = hoi_infos[:, 0]
#             ho_pairs = hoi_infos[:, 1:]
#         else:
#             action_probs = ho_pairs = ho_img_inds = None
#             obj_prob = None
#
#         if boxes_ext is not None:
#             boxes_ext = boxes_ext.cpu().numpy()
#             obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
#             obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
#             if obj_prob is None:
#                 obj_prob = boxes_ext[:, 5:]  # this cannot be refined because of the lack of spatial relationships
#         else:
#             obj_im_inds = obj_boxes = None
#         return Prediction(obj_im_inds=obj_im_inds,
#                           obj_boxes=obj_boxes,
#                           obj_scores=obj_prob,
#                           ho_img_inds=ho_img_inds,
#                           ho_pairs=ho_pairs,
#                           action_scores=action_probs)
#
#
# class ObjFGPredModel(GenericModel):
#     @classmethod
#     def get_cline_name(cls):
#         return 'objfgpred'
#
#     def __init__(self, dataset: HicoDetSplit, **kwargs):
#         self.fg_thr = 0.5
#         super().__init__(dataset, **kwargs)
#         vis_feat_dim = self.visual_module.vis_feat_dim
#
#         self.fg_obj_branch = SimpleObjBranch(input_dim=vis_feat_dim + self.dataset.num_object_classes)
#         self.fg_obj_output_fc = nn.Linear(self.fg_obj_branch.output_dim, 1)
#
#         self.obj_branch = SimpleObjBranch(input_dim=vis_feat_dim + self.dataset.num_object_classes)
#         self.obj_output_fc = nn.Linear(self.obj_branch.output_dim, self.dataset.num_object_classes)
#
#         self.act_branch = SimpleHoiBranch(self.visual_module.vis_feat_dim, self.obj_branch.output_dim, use_relu=cfg.model.relu)
#         self.action_output_fc = nn.Linear(self.act_branch.output_dim, dataset.num_predicates, bias=True)
#         torch.nn.init.xavier_normal_(self.action_output_fc.weight, gain=1.0)
#
#     def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
#         with torch.set_grad_enabled(self.training):
#             vis_output = self.visual_module(x, inference)  # type: VisualOutput
#
#             if vis_output.ho_infos is not None:
#                 obj_logits, action_logits, fg_obj_logits = self._forward(vis_output, )
#             else:
#                 obj_logits = action_logits = fg_obj_logits = None
#
#             if not inference:
#                 box_labels = vis_output.box_labels  # type: torch.Tensor
#                 action_labels = vis_output.action_labels
#                 fg_box_labels = torch.cat([box_labels.new_ones((box_labels.shape[0], 1)),
#                                            box_labels.new_zeros((fg_obj_logits.shape[0] - box_labels.shape[0], 1))], dim=0).float()
#
#                 losses = {'object_loss': nn.functional.cross_entropy(obj_logits, box_labels),
#                           'fg_object_loss': nn.functional.binary_cross_entropy_with_logits(fg_obj_logits, fg_box_labels),
#                           'action_loss': nn.functional.binary_cross_entropy_with_logits(action_logits, action_labels) * action_logits.shape[1]}
#                 return losses
#             else:
#                 prediction = Prediction()
#
#                 if vis_output.ho_infos is not None:
#                     assert action_logits is not None
#
#                     prediction.ho_img_inds = vis_output.ho_infos[:, 0]
#                     prediction.ho_pairs = vis_output.ho_infos[:, 1:]
#                     prediction.obj_scores = nn.functional.softmax(obj_logits, dim=1).cpu().numpy()
#                     prediction.action_scores = torch.sigmoid(action_logits).cpu().numpy()
#
#                 if vis_output.boxes_ext is not None:
#                     boxes_ext = vis_output.boxes_ext.cpu().numpy()
#                     im_scales = x.img_infos[:, 2].cpu().numpy()
#
#                     obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
#                     obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
#                     prediction.obj_im_inds = obj_im_inds
#                     prediction.obj_boxes = obj_boxes
#
#                 return prediction
#
#     def _forward(self, vis_output: VisualOutput, **kwargs):
#         if vis_output.box_labels is not None:
#             bg_boxes_ext, bg_box_feats, bg_masks = vis_output.filter_boxes()
#             boxes_ext = vis_output.boxes_ext
#             box_feats = vis_output.box_feats
#             masks = vis_output.masks
#
#             fg_and_bg_boxes_ext = torch.cat((boxes_ext, bg_boxes_ext), dim=0)
#             fg_and_bg_box_feats = torch.cat((box_feats, bg_box_feats), dim=0)
#
#             fg_obj_repr = self.fg_obj_branch(fg_and_bg_boxes_ext, fg_and_bg_box_feats)
#             fg_obj_logits = self.fg_obj_output_fc(fg_obj_repr)
#         else:
#             fg_and_bg_boxes_ext = vis_output.boxes_ext
#             fg_and_bg_box_feats = vis_output.box_feats
#
#             fg_obj_repr = self.fg_obj_branch(fg_and_bg_boxes_ext, fg_and_bg_box_feats)
#             fg_obj_logits = self.fg_obj_output_fc(fg_obj_repr)
#             fg_score = torch.sigmoid(fg_obj_logits.squeeze(dim=1))
#             vis_output.filter_boxes(valid_box_mask=(fg_score >= self.fg_thr))
#
#             if vis_output.boxes_ext is None:
#                 return None, None, None
#
#             boxes_ext = vis_output.boxes_ext
#             box_feats = vis_output.box_feats
#             masks = vis_output.masks
#
#         obj_repr = self.obj_branch(boxes_ext, box_feats)
#         obj_logits = self.obj_output_fc(obj_repr)
#
#         if vis_output.ho_infos is None:
#             assert vis_output.box_labels is None
#             return obj_logits, None, None
#
#         union_boxes_feats = vis_output.hoi_union_boxes_feats
#         hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)
#
#         act_repr = self.act_branch(obj_repr, union_boxes_feats, hoi_infos)
#         action_logits = self.action_output_fc(act_repr)
#
#         return obj_logits, action_logits, fg_obj_logits
#
#
# class FuncGenModel(GenericModel):
#     @classmethod
#     def get_cline_name(cls):
#         return 'fgen'
#
#     def __init__(self, dataset: HicoDetSplit, **kwargs):
#         self.fc1_dim = 1024
#         self.fc2_dim = 512
#         self.word_emb_dim = 300
#         super().__init__(dataset, **kwargs)
#         vis_feat_dim = self.visual_module.vis_feat_dim
#
#         word_embs = WordEmbeddings(source='glove', dim=self.word_emb_dim)
#         obj_word_embs = word_embs.get_embeddings(dataset.objects)
#         self.obj_word_embs = nn.Embedding.from_pretrained(torch.from_numpy(obj_word_embs), freeze=True)
#
#         self.act_branch = nn.Sequential(*[nn.Linear(vis_feat_dim + self.word_emb_dim + 14, self.fc1_dim),  # 14 = # geometric features
#                                           nn.ReLU(inplace=True),
#                                           nn.Linear(self.fc1_dim, self.fc2_dim),
#                                           nn.ReLU(inplace=True),
#                                           ])
#         self.act_output_fc = nn.Linear(self.fc2_dim, dataset.num_predicates, bias=True)
#
#     def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
#         with torch.set_grad_enabled(self.training):
#             vis_output = self.visual_module(x, inference)  # type: VisualOutput
#
#             if vis_output.ho_infos is not None:
#                 vis_output.minibatch = x
#                 action_output = self._forward(vis_output, )
#             else:
#                 assert inference
#                 action_output = None
#
#             if not inference:
#                 action_labels = vis_output.action_labels
#                 losses = {'action_loss': nn.functional.binary_cross_entropy_with_logits(action_output, action_labels) * action_output.shape[1]}
#                 return losses
#             else:
#                 prediction = Prediction()
#
#                 if vis_output.boxes_ext is not None:
#                     boxes_ext = vis_output.boxes_ext.cpu().numpy()
#                     im_scales = x.img_infos[:, 2].cpu().numpy()
#
#                     obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
#                     obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
#                     prediction.obj_im_inds = obj_im_inds
#                     prediction.obj_boxes = obj_boxes
#                     prediction.obj_scores = boxes_ext[:, 5:]
#
#                     if vis_output.ho_infos is not None:
#                         assert action_output is not None
#
#                         prediction.ho_img_inds = vis_output.ho_infos[:, 0]
#                         prediction.ho_pairs = vis_output.ho_infos[:, 1:]
#                         prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()
#
#                 return prediction
#
#     # noinspection PyMethodOverriding
#     def _forward(self, vis_output: VisualOutput, **kwargs):
#         if vis_output.box_labels is not None:
#             vis_output.filter_boxes()
#         boxes_ext = vis_output.boxes_ext
#         box_feats = vis_output.box_feats
#         masks = vis_output.masks
#         hoi_infos = torch.tensor(vis_output.ho_infos, device=masks.device)
#
#         im_sizes = torch.tensor(np.array([d['im_size'][::-1] * d['im_scale'] for d in batch.other_ex_data]).astype(np.float32), device=masks.device)
#         im_areas = im_sizes.prod(dim=1)
#
#         # Needed for numerical errors. Also when assigning GT to detections this is not guaranteed to be true.
#         # FIXME this should be needed, remove
#         box_im_inds = boxes_ext[:, 0].long()
#         box_im_sizes = im_sizes[box_im_inds, :]
#         boxes_ext[:, 3] = torch.min(boxes_ext[:, 3], box_im_sizes[:, 0])
#         boxes_ext[:, 4] = torch.min(boxes_ext[:, 4], box_im_sizes[:, 1])
#
#         norm_boxes = boxes_ext[:, 1:5] / box_im_sizes.repeat(1, 2)
#         assert (0 <= norm_boxes).all(), \
#             (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(),
#              im_sizes.detach().cpu().numpy(), norm_boxes.detach().cpu().numpy())
#         # norm_boxes.clamp_(max=1)  # Needed for numerical errors
#         assert (norm_boxes <= 1).all(), \
#             (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(),
#              im_sizes.detach().cpu().numpy(), norm_boxes.detach().cpu().numpy())
#
#         box_widths = boxes_ext[:, 3] - boxes_ext[:, 1]
#         box_heights = boxes_ext[:, 4] - boxes_ext[:, 2]
#         norm_box_areas = box_widths * box_heights / im_areas[box_im_inds]
#         assert (0 < norm_box_areas).all(), \
#             (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(), norm_box_areas.detach().cpu().numpy())
#         norm_box_areas.clamp_(max=1)  # Same as above
#         assert (norm_box_areas <= 1).all(), \
#             (box_im_inds.detach().cpu().numpy(), boxes_ext[:, 1:5].detach().cpu().numpy(), norm_box_areas.detach().cpu().numpy())
#
#         hum_inds = hoi_infos[:, 1]
#         obj_inds = hoi_infos[:, 2]
#         obj_widths = box_widths[obj_inds]
#         obj_heights = box_widths[obj_inds]
#
#         h_dist = (boxes_ext[hum_inds, 1] - boxes_ext[obj_inds, 1]) / obj_widths
#         v_dist = (boxes_ext[hum_inds, 2] - boxes_ext[obj_inds, 2]) / obj_heights
#
#         h_ratio = (box_widths[hum_inds] / obj_widths).log()
#         v_ratio = (box_widths[hum_inds] / obj_heights).log()
#
#         geo_feats = torch.cat([norm_boxes[hum_inds, :],
#                                norm_box_areas[hum_inds, None],
#                                norm_boxes[obj_inds, :],
#                                norm_box_areas[obj_inds, None],
#                                h_dist[:, None], v_dist[:, None], h_ratio[:, None], v_ratio[:, None]
#                                ], dim=1)
#
#         obj_word_embs = self.obj_word_embs(boxes_ext[obj_inds, 5:].argmax(dim=1))
#
#         hum_repr = box_feats[hum_inds, :]
#
#         act_repr = self.act_branch(torch.cat([geo_feats, obj_word_embs, hum_repr], dim=1))
#         action_logits = self.act_output_fc(act_repr)
#
#         return action_logits
#
# class ZSHoiModel(ZSBaseModel):
#     @classmethod
#     def get_cline_name(cls):
#         return 'zsh'
#
#     def __init__(self, dataset: HicoDetSplit, **kwargs):
#         self.emb_dim = 200
#         super().__init__(dataset, **kwargs)
#
#         latent_dim = self.emb_dim
#         input_dim = self.predictor_dim
#         self.vrepr_to_emb = nn.Sequential(*[nn.Linear(input_dim, 800),
#                                             nn.ReLU(inplace=True),
#                                             nn.Dropout(p=cfg.model.dropout),
#                                             nn.Linear(800, 600),
#                                             nn.ReLU(inplace=True),
#                                             nn.Dropout(p=cfg.model.dropout),
#                                             nn.Linear(600, 2 * latent_dim),
#                                             ])
#         self.emb_to_predictor = nn.Sequential(*[nn.Linear(latent_dim, 600),
#                                                 nn.ReLU(inplace=True),
#                                                 nn.Dropout(p=cfg.model.dropout),
#                                                 nn.Linear(600, 800),
#                                                 nn.ReLU(inplace=True),
#                                                 nn.Dropout(p=cfg.model.dropout),
#                                                 nn.Linear(800, input_dim),
#                                                 ])
#
#         self.gcn = CheatHoiGCNBranch(dataset, input_repr_dim=512, gc_dims=(300, self.emb_dim))
#         adj_av = (self.gcn.adj_av > 0).float()
#         self.adj_av_norm = nn.Parameter(adj_av / adj_av.sum(dim=1, keepdim=True))
#
#         if cfg.model.softl:
#             self.obj_act_feasibility = nn.Parameter(self.gcn.noun_verb_links, requires_grad=False)
#
#     def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):
#         vrepr = self.base_model._forward(vis_output, return_repr=True)
#
#         hoi_class_embs, _, _ = self.gcn()  # I x E
#         if vis_output.action_labels is not None:
#             action_labels = vis_output.action_labels
#         else:
#             action_labels = None
#
#         if cfg.model.attw:
#             instance_params = self.vrepr_to_emb(vrepr)
#             instance_means = instance_params[:, :instance_params.shape[1] // 2]  # P x E
#             instance_logstd = instance_params[:, instance_params.shape[1] // 2:]  # P x E
#             instance_logstd = instance_logstd.unsqueeze(dim=1)
#             instance_class_logprobs = - 0.5 * (2 * instance_logstd.sum(dim=2) +  # NOTE: constant term is missing
#                                                ((instance_means.unsqueeze(dim=1) - hoi_class_embs.unsqueeze(dim=0)) / instance_logstd.exp()).norm(
#                                                    dim=2) ** 2)
#             hoi_predictors = self.emb_to_predictor(instance_class_logprobs.exp().unsqueeze(dim=2) *
#                                                    F.normalize(hoi_class_embs, dim=1).unsqueeze(dim=0))  # N x P x D
#             hoi_logits = torch.bmm(vrepr.unsqueeze(dim=1), hoi_predictors.transpose(1, 2)).squeeze(dim=1)
#         else:
#             hoi_predictors = self.emb_to_predictor(hoi_class_embs)  # P x D
#             hoi_logits = vrepr @ hoi_predictors.t()
#
#         ho_obj_inter_prior = vis_output.boxes_ext[vis_output.ho_infos_np[:, 2], 5:][:, self.dataset.hicodet.interactions[:, 1]]
#         hoi_logits = hoi_logits * ho_obj_inter_prior
#
#         act_logits = hoi_logits @ self.adj_av_norm  # get action class embeddings through marginalisation
#
#         if vis_output.action_labels is not None and not cfg.model.softl:  # restrict training to seen predicates only
#             act_logits = act_logits[:, self.seen_pred_inds]
#
#         reg_loss = None
#         return act_logits, action_labels, reg_loss
#
#
# class ZSObjModel(ZSBaseModel):
#     @classmethod
#     def get_cline_name(cls):
#         return 'zso'
#
#     def __init__(self, dataset: HicoDetSplit, **kwargs):
#         self.emb_dim = 200
#         super().__init__(dataset, **kwargs)
#
#         latent_dim = self.emb_dim
#         input_dim = self.predictor_dim
#         self.vrepr_to_emb = nn.Sequential(*[nn.Linear(input_dim, 800),
#                                             nn.ReLU(inplace=True),
#                                             nn.Dropout(p=cfg.model.dropout),
#                                             nn.Linear(800, 600),
#                                             nn.ReLU(inplace=True),
#                                             nn.Dropout(p=cfg.model.dropout),
#                                             nn.Linear(600, 2 * latent_dim),
#                                             ])
#         self.emb_to_predictor = nn.Sequential(*[nn.Linear(latent_dim, 600),
#                                                 nn.ReLU(inplace=True),
#                                                 nn.Dropout(p=cfg.model.dropout),
#                                                 nn.Linear(600, 800),
#                                                 nn.ReLU(inplace=True),
#                                                 nn.Dropout(p=cfg.model.dropout),
#                                                 nn.Linear(800, input_dim),
#                                                 ])
#         self.emb_to_obj_predictor = nn.Sequential(*[nn.Linear(latent_dim, 600),
#                                                     nn.ReLU(inplace=True),
#                                                     nn.Dropout(p=cfg.model.dropout),
#                                                     nn.Linear(600, 800),
#                                                     nn.ReLU(inplace=True),
#                                                     nn.Dropout(p=cfg.model.dropout),
#                                                     nn.Linear(800, input_dim),
#                                                     ])
#
#         self.gcn = CheatGCNBranch(dataset, input_repr_dim=512, gc_dims=(300, self.emb_dim))
#
#         if cfg.model.softl:
#             self.obj_act_feasibility = nn.Parameter(self.gcn.noun_verb_links, requires_grad=False)
#
#     def get_soft_labels(self, vis_output: VisualOutput):
#         ho_infos = torch.tensor(vis_output.ho_infos_np, device=vis_output.action_labels.device)
#         action_labels = vis_output.boxes_ext[ho_infos[:, 2], 5:] @ self.obj_act_feasibility
#
#         action_labels[:, self.seen_pred_inds] = vis_output.action_labels
#
#         action_labels = action_labels.detach()
#         return action_labels
#
#     def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
#         with torch.set_grad_enabled(self.training):
#             vis_output = self.visual_module(x, inference)  # type: VisualOutput
#
#             if vis_output.ho_infos_np is not None:
#                 action_output, action_labels, reg_loss, ho_obj_output, ho_obj_labels = self._forward(vis_output, epoch=x.epoch, step=x.iter)
#                 if inference and self.load_backbone:
#                     pretrained_vrepr = self.pretrained_base_model._forward(vis_output, return_repr=True).detach()
#                     pretrained_action_output = pretrained_vrepr @ self.pretrained_predictors.t()  # N x Pt
#
#                     action_output[:, self.seen_pred_inds] = pretrained_action_output
#             else:
#                 assert inference
#                 action_output = action_labels = None
#
#             if not inference:
#                 losses = {'action_loss': F.binary_cross_entropy_with_logits(action_output, action_labels) * action_output.shape[1],
#                           'obj_loss': F.cross_entropy(ho_obj_output, ho_obj_labels),
#                           }
#                 if reg_loss is not None:
#                     losses['reg_loss'] = reg_loss
#                 return losses
#             else:
#                 prediction = Prediction()
#
#                 if vis_output.boxes_ext is not None:
#                     boxes_ext = vis_output.boxes_ext.cpu().numpy()
#                     im_scales = x.img_infos[:, 2]
#
#                     obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
#                     obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
#                     prediction.obj_im_inds = obj_im_inds
#                     prediction.obj_boxes = obj_boxes
#                     prediction.obj_scores = boxes_ext[:, 5:]
#
#                     if vis_output.ho_infos_np is not None:
#                         assert action_output is not None
#
#                         prediction.ho_img_inds = vis_output.ho_infos_np[:, 0]
#                         prediction.ho_pairs = vis_output.ho_infos_np[:, 1:]
#
#                         prediction.action_scores = torch.sigmoid(action_output).cpu().numpy()
#
#                 return prediction
#
#     def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):
#         act_vrepr, ho_obj_vrepr = self.base_model._forward(vis_output, return_repr=True, return_obj=True)
#         act_instance_params = self.vrepr_to_emb(act_vrepr)
#         act_instance_means = act_instance_params[:, :act_instance_params.shape[1] // 2]  # P x E
#         act_instance_logstd = act_instance_params[:, act_instance_params.shape[1] // 2:]  # P x E
#
#         obj_class_embs, act_class_embs = self.gcn()  # P x E
#         if vis_output.action_labels is not None:
#             obj_predictors = self.emb_to_obj_predictor(obj_class_embs)  # P x D
#             ho_obj_output = ho_obj_vrepr @ obj_predictors.t()
#             ho_infos = torch.tensor(vis_output.ho_infos_np, device=ho_obj_vrepr.device)
#             ho_obj_labels = vis_output.box_labels[ho_infos[:, 2]]
#             fg_objs = (ho_obj_labels >= 0)
#             ho_obj_output = ho_obj_output[fg_objs, :]
#             ho_obj_labels = ho_obj_labels[fg_objs]
#
#             act_class_embs = act_class_embs[self.seen_pred_inds, :]  # P x E
#             action_labels = vis_output.action_labels
#         else:
#             action_labels = ho_obj_output = ho_obj_labels = None
#
#         act_instance_logstd = act_instance_logstd.unsqueeze(dim=1)
#         class_logprobs = - 0.5 * (2 * act_instance_logstd.sum(dim=2) +  # NOTE: constant term is missing
#                                   ((act_instance_means.unsqueeze(dim=1) - act_class_embs.unsqueeze(dim=0)) /
#                                    act_instance_logstd.exp()).norm(dim=2) ** 2)
#
#         if cfg.model.attw:
#             act_predictors = self.emb_to_predictor(class_logprobs.exp().unsqueeze(dim=2) *
#                                                    F.normalize(act_class_embs, dim=1).unsqueeze(dim=0))  # N x P x D
#             action_output = torch.bmm(act_vrepr.unsqueeze(dim=1), act_predictors.transpose(1, 2)).squeeze(dim=1)
#         else:
#             act_predictors = self.emb_to_predictor(act_class_embs)  # P x D
#             action_output = act_vrepr @ act_predictors.t()
#
#         reg_loss = None
#         return action_output, action_labels, reg_loss, ho_obj_output, ho_obj_labels
#
#
# class ZSGCModel(ZSGenericModel):
#     @classmethod
#     def get_cline_name(cls):
#         return 'zsgc'
#
#     def __init__(self, dataset: HicoDetSplit, **kwargs):
#         super().__init__(dataset, **kwargs)
#         self.base_model = BaseModel(dataset, **kwargs)
#         self.predictor_dim = self.base_model.final_repr_dim
#
#         if cfg.model.large:
#             gcemb_dim = 2048
#             self.emb_dim = 512
#         else:
#             gcemb_dim = 1024
#             self.emb_dim = 200
#
#         latent_dim = self.emb_dim
#         input_dim = self.predictor_dim
#         self.vrepr_to_emb = nn.Sequential(*[nn.Linear(input_dim, 800),
#                                             nn.ReLU(inplace=True),
#                                             nn.Dropout(p=cfg.model.dropout),
#                                             nn.Linear(800, 600),
#                                             nn.ReLU(inplace=True),
#                                             nn.Dropout(p=cfg.model.dropout),
#                                             nn.Linear(600, 2 * latent_dim),
#                                             ])
#         self.emb_to_predictor = nn.Sequential(*[nn.Linear(latent_dim, 600),
#                                                 nn.ReLU(inplace=True),
#                                                 nn.Dropout(p=cfg.model.dropout),
#                                                 nn.Linear(600, 800),
#                                                 nn.ReLU(inplace=True),
#                                                 nn.Dropout(p=cfg.model.dropout),
#                                                 nn.Linear(800, input_dim),
#                                                 ])
#
#         if cfg.model.oscore:
#             self.obj_scores_to_act_logits = nn.Sequential(*[nn.Linear(self.dataset.num_object_classes, self.dataset.hicodet.num_predicates)])
#
#         if cfg.model.vv:
#             assert not cfg.model.iso_null, 'Not supported'
#             self.gcn = ExtCheatGCNBranch(dataset, input_repr_dim=gcemb_dim, gc_dims=(gcemb_dim // 2, self.emb_dim))
#         else:
#             self.gcn = CheatGCNBranch(dataset, input_repr_dim=gcemb_dim, gc_dims=(gcemb_dim // 2, self.emb_dim))
#
#         if cfg.model.aereg > 0:
#             if cfg.model.regsmall:
#                 self.emb_to_wemb = nn.Linear(latent_dim, self.pred_word_embs.shape[1])
#             else:
#                 self.emb_to_wemb = nn.Sequential(*[nn.Linear(latent_dim, latent_dim),
#                                                    nn.ReLU(inplace=True),
#                                                    nn.Linear(latent_dim, self.pred_word_embs.shape[1]),
#                                                    ])
#
#         op_mat = np.zeros([dataset.hicodet.num_object_classes, dataset.hicodet.num_predicates], dtype=np.float32)
#         for _, p, o in dataset.hoi_triplets:
#             op_mat[o, p] += 1
#         op_mat /= np.maximum(1, op_mat.sum(axis=1, keepdims=True))
#         self.op_mat = nn.Parameter(torch.from_numpy(op_mat)[:, self.unseen_pred_inds], requires_grad=False)
#
#         if cfg.model.softl:
#             self.obj_act_feasibility = nn.Parameter(self.gcn.noun_verb_links, requires_grad=False)
#
#     def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):
#         vrepr = self.base_model._forward(vis_output, return_repr=True)
#
#         _, all_class_embs = self.gcn()  # P x E
#         class_embs = all_class_embs
#         action_labels = vis_output.action_labels
#         unseen_action_labels = None
#         if action_labels is not None:
#             if cfg.model.softl > 0:
#                 unseen_action_labels = self.get_soft_labels(vis_output)
#             else:  # restrict training to seen predicates only
#                 class_embs = all_class_embs[self.seen_pred_inds, :]  # P x E
#
#         if cfg.model.attw:
#             instance_params = self.vrepr_to_emb(vrepr)
#             instance_means = instance_params[:, :instance_params.shape[1] // 2]  # P x E
#             instance_logstd = instance_params[:, instance_params.shape[1] // 2:]  # P x E
#             instance_logstd = instance_logstd.unsqueeze(dim=1)
#             class_logprobs = - 0.5 * (2 * instance_logstd.sum(dim=2) +  # NOTE: constant term is missing
#                                       ((instance_means.unsqueeze(dim=1) - class_embs.unsqueeze(dim=0)) / instance_logstd.exp()).norm(dim=2) ** 2)
#             act_predictors = self.emb_to_predictor(class_logprobs.exp().unsqueeze(dim=2) *
#                                                    F.normalize(class_embs, dim=1).unsqueeze(dim=0))  # N x P x D
#             action_logits = torch.bmm(vrepr.unsqueeze(dim=1), act_predictors.transpose(1, 2)).squeeze(dim=1)
#         else:
#             act_predictors = self.emb_to_predictor(class_embs)  # P x D
#             action_logits = vrepr @ act_predictors.t()
#
#         if cfg.model.oscore:
#             action_logits_from_obj_score = self.obj_scores_to_act_logits(vis_output.boxes_ext[vis_output.ho_infos_np[:, 2], 5:])
#             if vis_output.action_labels is not None and not cfg.model.softl:
#                 action_logits_from_obj_score = action_logits_from_obj_score[:, self.seen_pred_inds]  # P x E
#             action_logits = action_logits + action_logits_from_obj_score
#
#         if cfg.model.aereg > 0:
#             reg_loss = -cfg.model.aereg * (F.normalize(self.emb_to_wemb(all_class_embs)) * self.pred_word_embs).sum(dim=1).mean()
#         else:
#             reg_loss = None
#         return action_logits, action_labels, reg_loss, unseen_action_labels
#
# class BGFilter(BaseModel):
#     @classmethod
#     def get_cline_name(cls):
#         return 'bg'
#
#     def __init__(self, dataset: HicoDetSplit, **kwargs):
#         super().__init__(dataset, **kwargs)
#         self.bg_vis_mlp = nn.Sequential(*[nn.Linear(self.visual_module.vis_feat_dim, 1024),
#                                           nn.ReLU(inplace=True),
#                                           nn.Dropout(p=cfg.model.dropout),
#                                           nn.Linear(1024, 512),
#                                           ])
#         # self.bg_geo_obj_mlp = nn.Sequential(*[nn.Linear(14 + self.dataset.num_object_classes, 128),
#         #                                       nn.Linear(128, 256),
#         #                                       ])
#         #
#         # self.bg_detection_mlp = nn.Linear(512 + 256, 1, bias=False)
#         self.bg_detection_mlp = nn.Linear(512, 1, bias=False)
#         torch.nn.init.xavier_normal_(self.bg_detection_mlp.weight, gain=1.0)
#
#     def forward(self, x: PrecomputedMinibatch, inference=True, **kwargs):
#         with torch.set_grad_enabled(self.training):
#             vis_output = self.visual_module(x, inference)  # type: VisualOutput
#
#             if vis_output.ho_infos_np is not None:
#                 action_output, bg_output = self._forward(vis_output, batch=x, epoch=x.epoch, step=x.iter)
#             else:
#                 assert inference
#                 action_output = bg_output = None
#
#             if not inference:
#                 action_labels = vis_output.action_labels
#                 if cfg.data.null_as_bg:
#                     bg_label = action_labels[:, 0]
#                     max_fg_score = torch.sigmoid((action_output * action_labels)[:, 1:].max(dim=1)[0]).detach()
#                     act_loss = F.binary_cross_entropy_with_logits(action_output[:, 1:], action_labels[:, 1:]) * (action_output.shape[1] - 1)
#                 else:
#                     bg_label = 1 - (action_labels > 0).any(dim=1).float()
#                     max_fg_score = torch.sigmoid((action_output * action_labels).max(dim=1)[0]).detach()
#                     act_loss = F.binary_cross_entropy_with_logits(action_output, action_labels) * action_output.shape[1]
#
#                 losses = {'action_loss': act_loss}
#                 if cfg.opt.margin > 0:
#                     bg_score = torch.sigmoid(bg_output).squeeze(dim=1)
#                     bg_loss = F.margin_ranking_loss(bg_score, max_fg_score, 2 * bg_label - 1, margin=cfg.opt.margin, reduction='none').mean()
#                 else:
#                     bg_loss = F.binary_cross_entropy_with_logits(bg_output, bg_label[:, None])
#                 losses['bg_loss'] = cfg.opt.bg_coeff * bg_loss
#                 return losses
#             else:
#                 prediction = Prediction()
#
#                 if vis_output.boxes_ext is not None:
#                     boxes_ext = vis_output.boxes_ext.cpu().numpy()
#                     im_scales = x.img_infos[:, 2]
#
#                     obj_im_inds = boxes_ext[:, 0].astype(np.int, copy=False)
#                     obj_boxes = boxes_ext[:, 1:5] / im_scales[obj_im_inds, None]
#                     prediction.obj_im_inds = obj_im_inds
#                     prediction.obj_boxes = obj_boxes
#                     prediction.obj_scores = boxes_ext[:, 5:]
#
#                     if vis_output.ho_infos_np is not None:
#                         assert action_output is not None
#
#                         if action_output.shape[1] < self.dataset.hicodet.num_predicates:
#                             assert action_output.shape[1] == self.dataset.num_predicates
#                             restricted_action_output = action_output
#                             action_output = restricted_action_output.new_zeros((action_output.shape[0], self.dataset.hicodet.num_predicates))
#                             action_output[:, self.dataset.active_predicates] = restricted_action_output
#
#                         action_scores = torch.sigmoid(action_output).cpu().numpy()
#                         bg_scores = torch.sigmoid(bg_output).squeeze(dim=1).cpu().numpy()
#                         action_scores[:, 0] = bg_scores
#
#                         if cfg.model.filter:
#                             # keep = (action_scores[:, 1:].max(axis=1) > bg_scores)
#                             keep = (bg_scores < 0.95)
#
#                             if np.any(keep):
#                                 prediction.ho_img_inds = vis_output.ho_infos_np[keep, 0]
#                                 prediction.ho_pairs = vis_output.ho_infos_np[keep, 1:]
#                                 prediction.action_scores = action_scores[keep, :]
#                         else:
#                             action_scores[:, 1:] *= (1 - bg_scores[:, None])
#                             prediction.ho_img_inds = vis_output.ho_infos_np[:, 0]
#                             prediction.ho_pairs = vis_output.ho_infos_np[:, 1:]
#                             prediction.action_scores = action_scores
#
#                 return prediction
#
#     def _forward(self, vis_output: VisualOutput, batch=None, step=None, epoch=None, **kwargs):
#         boxes_ext = vis_output.boxes_ext
#         ho_infos = vis_output.ho_infos
#         union_boxes_feats = vis_output.hoi_union_boxes_feats
#
#         # geo_feats = self.get_geo_feats(vis_output, batch)
#
#         vis_repr = self.bg_vis_mlp(union_boxes_feats)
#         # geo_obj_repr = self.bg_geo_obj_mlp(torch.cat([geo_feats, boxes_ext[ho_infos[:, 2], 5:]], dim=1))
#
#         # bg_logits = self.bg_detection_mlp(torch.cat([vis_repr, geo_obj_repr], dim=1))
#         bg_logits = self.bg_detection_mlp(vis_repr)
#
#         action_logits = super()._forward(vis_output, batch, step, epoch, **kwargs)
#         return action_logits, bg_logits

