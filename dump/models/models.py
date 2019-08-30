#
# class MultiModel(GenericModel):
#     @classmethod
#     def get_cline_name(cls):
#         return 'multi'
#
#     def __init__(self, dataset: HicoDetSplit, **kwargs):
#         super().__init__(dataset, **kwargs)
#         vis_feat_dim = self.visual_module.vis_feat_dim
#         hidden_dim = 1024
#         self.output_repr_dim = cfg.repr_dim
#
#         self.hoi_subj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, hidden_dim),
#                                                  nn.ReLU(inplace=True),
#                                                  nn.Dropout(p=cfg.dropout),
#                                                  nn.Linear(hidden_dim, self.final_repr_dim),
#                                                  ])
#         self.hoi_obj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, hidden_dim),
#                                                 nn.ReLU(inplace=True),
#                                                 nn.Dropout(p=cfg.dropout),
#                                                 nn.Linear(hidden_dim, self.final_repr_dim),
#                                                 ])
#         self.hoi_act_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, hidden_dim),
#                                                 nn.ReLU(inplace=True),
#                                                 nn.Dropout(p=cfg.dropout),
#                                                 nn.Linear(hidden_dim, self.final_repr_dim),
#                                                 ])
#
#         self.act_subj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, hidden_dim),
#                                                  nn.ReLU(inplace=True),
#                                                  nn.Dropout(p=cfg.dropout),
#                                                  nn.Linear(hidden_dim, self.final_repr_dim),
#                                                  ])
#         self.act_obj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, hidden_dim),
#                                                 nn.ReLU(inplace=True),
#                                                 nn.Dropout(p=cfg.dropout),
#                                                 nn.Linear(hidden_dim, self.final_repr_dim),
#                                                 ])
#         self.act_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim, hidden_dim),
#                                             nn.ReLU(inplace=True),
#                                             nn.Dropout(p=cfg.dropout),
#                                             nn.Linear(hidden_dim, self.final_repr_dim),
#                                             ])
#
#         self.act_output_mlp = nn.Linear(self.final_repr_dim, dataset.num_actions, bias=False)
#         torch.nn.init.xavier_normal_(self.act_output_mlp.weight, gain=1.0)
#         self.hoi_output_mlp = nn.Linear(self.final_repr_dim, dataset.full_dataset.num_interactions, bias=False)
#         torch.nn.init.xavier_normal_(self.hoi_output_mlp.weight, gain=1.0)
#
#     def _get_losses(self, vis_output: VisualOutput, outputs):
#         act_output, hoi_output = outputs
#         losses = {'hoi_loss': bce_loss(hoi_output, vis_output.hoi_labels),
#                   'action_loss': bce_loss(act_output, vis_output.action_labels)}
#         return losses
#
#     def _finalize_prediction(self, prediction: Prediction, vis_output: VisualOutput, outputs):
#         act_output, hoi_output = outputs
#
#         if act_output.shape[1] < self.dataset.full_dataset.num_actions:
#             assert act_output.shape[1] == self.dataset.num_actions
#             restricted_action_output = act_output
#             act_output = restricted_action_output.new_zeros((act_output.shape[0], self.dataset.full_dataset.num_actions))
#             act_output[:, self.dataset.active_predicates] = restricted_action_output
#         prediction.action_scores = torch.sigmoid(act_output).cpu().numpy()
#
#         ho_obj_scores = prediction.obj_scores[vis_output.ho_infos_np[:, 2], :]
#         hoi_obj_scores = ho_obj_scores[:, self.dataset.full_dataset.interactions[:, 1]]  # This helps
#         prediction.hoi_scores = torch.sigmoid(hoi_output).cpu().numpy() * \
#                                 hoi_obj_scores * \
#                                 prediction.action_scores[:, self.dataset.full_dataset.interactions[:, 0]]
#
#     @property
#     def final_repr_dim(self):
#         return self.output_repr_dim
#
#     def _forward(self, vis_output: VisualOutput, batch=None, step=None, epoch=None):
#         boxes_ext = vis_output.boxes_ext
#         box_feats = vis_output.box_feats
#         hoi_infos = vis_output.ho_infos
#         union_boxes_feats = vis_output.hoi_union_boxes_feats
#
#         subj_ho_feats = torch.cat([box_feats[hoi_infos[:, 1], :], boxes_ext[hoi_infos[:, 1], 5:]], dim=1)
#         obj_ho_feats = torch.cat([box_feats[hoi_infos[:, 2], :], boxes_ext[hoi_infos[:, 2], 5:]], dim=1)
#
#         hoi_subj_repr = self.hoi_subj_repr_mlp(subj_ho_feats)
#         hoi_obj_repr = self.hoi_obj_repr_mlp(obj_ho_feats)
#         hoi_act_repr = self.hoi_act_repr_mlp(union_boxes_feats)
#         hoi_logits = self.hoi_output_mlp(hoi_subj_repr + hoi_obj_repr + hoi_act_repr)
#
#         act_subj_repr = self.act_subj_repr_mlp(subj_ho_feats)
#         act_obj_repr = self.act_obj_repr_mlp(obj_ho_feats)
#         act_repr = self.act_repr_mlp(union_boxes_feats)
#         act_logits = self.act_output_mlp(act_subj_repr + act_obj_repr + act_repr)
#
#         return act_logits, hoi_logits
#
# class ZSSimModel(ZSBaseModel):
#     @classmethod
#     def get_cline_name(cls):
#         return 'zss'
#
#     def __init__(self, dataset: HicoDetSplit, **kwargs):
#         super().__init__(dataset, **kwargs)
#         assert cfg.softl > 0
#
#         seen_pred_inds = pickle.load(open(cfg.active_classes_file, 'rb'))[Splits.TRAIN.value]['pred']
#         seen_transfer_pred_inds = pickle.load(open(cfg.active_classes_file, 'rb'))[Splits.TRAIN.value]['pred_transfer']
#         seen_train_pred_inds = np.array([p for p in seen_pred_inds if p not in seen_transfer_pred_inds])
#         all_transfer_pred_inds = np.array(sorted(set(range(self.dataset.full_dataset.num_actions)) - set(seen_train_pred_inds.tolist())))
#         self.seen_train_inds = nn.Parameter(torch.from_numpy(seen_train_pred_inds), requires_grad=False)
#         self.seen_transfer_inds = nn.Parameter(torch.from_numpy(seen_transfer_pred_inds), requires_grad=False)
#         self.all_transfer_inds = nn.Parameter(torch.from_numpy(all_transfer_pred_inds), requires_grad=False)
#
#         # these are RELATIVE to the seen ones
#         rel_seen_transfer_pred_inds = sorted([np.flatnonzero(seen_pred_inds == p)[0] for p in seen_transfer_pred_inds])
#         rel_seen_train_pred_inds = sorted(set(range(len(seen_pred_inds))) - set(rel_seen_transfer_pred_inds))
#         self.rel_seen_transfer_inds = nn.Parameter(torch.from_numpy(np.array(rel_seen_transfer_pred_inds)), requires_grad=False)
#         self.rel_seen_train_inds = nn.Parameter(torch.from_numpy(np.array(rel_seen_train_pred_inds)), requires_grad=False)
#
#         wemb_dim = self.pred_word_embs.shape[1]
#         self.soft_labels_emb_mlp = nn.Sequential(*[nn.Linear(wemb_dim * 2, wemb_dim * 2),
#                                                    nn.ReLU(inplace=True),
#                                                    # nn.Dropout(p=cfg.dropout),
#                                                    nn.Linear(wemb_dim * 2, wemb_dim),
#                                                    ])
#
#     def get_soft_labels(self, vis_output: VisualOutput):
#         # unseen_action_labels = self.obj_act_feasibility[:, self.unseen_pred_inds][vis_output.box_labels[vis_output.ho_infos_np[:, 2]], :] * 0.75
#         # unseen_action_labels = vis_output.boxes_ext[vis_output.ho_infos_np[:, 2], 5:] @ self.obj_act_feasibility[:, self.unseen_pred_inds]
#         # unseen_action_labels = self.op_mat[vis_output.box_labels[vis_output.ho_infos_np[:, 2]], :]
#
#         obj_labels = vis_output.box_labels[vis_output.ho_infos_np[:, 2]]
#         bg_objs_mask = obj_labels < 0
#
#         known_labels = vis_output.action_labels
#         train_labels = known_labels[:, self.rel_seen_train_inds]
#         obj_embs = self.obj_word_embs[obj_labels, :]
#         obj_embs[bg_objs_mask, :] = 0
#         coocc_act_embs_avg = train_labels @ self.pred_word_embs[self.seen_train_inds, :] / train_labels.sum(dim=1, keepdim=True).clamp(min=1)
#         unseen_action_embs = self.soft_labels_emb_mlp(torch.cat([obj_embs, coocc_act_embs_avg], dim=1))
#
#         # these are for ALL actions
#         action_labels_mask = self.obj_act_feasibility[obj_labels, :]
#         action_labels_mask[bg_objs_mask, :] = 0
#         surrogate_action_labels = (F.normalize(unseen_action_embs, dim=1) @ self.pred_word_embs.t()) * action_labels_mask
#         surrogate_action_labels.clamp_(min=0, max=1)
#         if cfg.lis:
#             surrogate_action_labels = LIS(surrogate_action_labels, w=18, k=7)
#
#         # Loss is for transfer only, actual labels for unseen only
#         transfer_labels = known_labels[:, self.rel_seen_transfer_inds].detach()
#         unseen_action_label_loss = F.binary_cross_entropy(surrogate_action_labels[:, self.seen_transfer_inds], transfer_labels)  # Correct! No logits!
#
#         unseen_action_labels = surrogate_action_labels[:, self.unseen_pred_inds]
#         return unseen_action_labels.detach(), unseen_action_label_loss
#
#     def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):
#         vrepr = self.base_model._forward(vis_output, return_repr=True)
#
#         action_labels = vis_output.action_labels
#         unseen_action_labels, unseen_action_label_loss = None, None
#         if action_labels is not None:
#             unseen_action_labels, unseen_action_label_loss = self.get_soft_labels(vis_output)
#
#         reg_loss = unseen_action_label_loss  # FIXME hack
#         action_logits = self.output_mlp(vrepr)
#         return action_logits, action_labels, reg_loss, unseen_action_labels
#
#
# class KatoModel(GenericModel):
#     @classmethod
#     def get_cline_name(cls):
#         return 'kato'
#
#     def __init__(self, dataset: HicoDetSplit, **kwargs):
#         super().__init__(dataset, **kwargs)
#         vis_feat_dim = self.visual_module.vis_feat_dim
#         self.hoi_repr_dim = 600
#
#         self.ho_subj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.final_repr_dim),
#                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                 nn.Dropout(p=cfg.dropout),
#                                                 nn.Linear(self.final_repr_dim, self.final_repr_dim),
#                                                 ])
#         nn.init.xavier_normal_(self.ho_subj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
#
#         self.ho_obj_repr_mlp = nn.Sequential(*[nn.Linear(vis_feat_dim + self.dataset.num_object_classes, self.final_repr_dim),
#                                                nn.LeakyReLU(negative_slope=0.2, inplace=True),
#                                                nn.Dropout(p=cfg.dropout),
#                                                nn.Linear(self.final_repr_dim, self.final_repr_dim),
#                                                ])
#         nn.init.xavier_normal_(self.ho_obj_repr_mlp[0].weight, gain=torch.nn.init.calculate_gain('leaky_relu'))
#
#         gc_dims = (512, 200)
#         self.gcn_branch = KatoGCNBranch(dataset, gc_dims=(512, 200))
#         self.score_mlp = nn.Sequential(nn.Linear(gc_dims[-1] + self.hoi_repr_dim, 512),
#                                        nn.ReLU(inplace=True),
#                                        nn.Dropout(p=0.5),
#                                        nn.Linear(512, 200),
#                                        nn.ReLU(inplace=True),
#                                        nn.Dropout(p=0.5),
#                                        nn.Linear(200, 1)
#                                        )
#
#     @property
#     def final_repr_dim(self):
#         return self.hoi_repr_dim
#
#     def _get_losses(self, vis_output: VisualOutput, outputs):
#         hoi_output = outputs
#         hoi_labels = vis_output.hoi_labels
#         losses = {'hoi_loss': bce_loss(hoi_output, hoi_labels)}
#         return losses
#
#     def _finalize_prediction(self, prediction: Prediction, vis_output: VisualOutput, outputs):
#         hoi_output = outputs
#         prediction.hoi_scores = torch.sigmoid(hoi_output).cpu().numpy()
#
#     def _forward(self, vis_output: VisualOutput, step=None, epoch=None, **kwargs):
#         boxes_ext = vis_output.boxes_ext
#         box_feats = vis_output.box_feats
#         hoi_infos = vis_output.ho_infos
#
#         box_feats_ext = torch.cat([box_feats, boxes_ext[:, 5:]], dim=1)
#
#         ho_subj_repr = self.ho_subj_repr_mlp(box_feats_ext[hoi_infos[:, 1], :])
#         ho_obj_repr = self.ho_obj_repr_mlp(box_feats_ext[hoi_infos[:, 2], :])
#         hoi_repr = ho_subj_repr + ho_obj_repr
#
#         z_a, z_v, z_n = self.gcn_branch()
#         hoi_logits = self.score_mlp(torch.cat([hoi_repr.unsqueeze(dim=1).expand(-1, z_a.shape[0], -1),
#                                                z_a.unsqueeze(dim=0).expand(hoi_repr.shape[0], -1, -1)],
#                                               dim=2))
#         assert hoi_logits.shape[2] == 1
#         hoi_logits = hoi_logits.squeeze(dim=2)
#         return hoi_logits