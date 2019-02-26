import torch
import torch.nn as nn


class SpatialContext(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()

        # FIXME params
        self.use_bn = False  # Since the batches are small due to memory constraint, BN is not suitable. TODO Maybe switch to GN?
        self.spatial_dropout = 0.1
        self.spatial_emb_dim = 64
        self.spatial_rnn_emb_dim = 64
        self.spatial_rnn_dropout = 0.1

        self.spatial_rels_fc = nn.Sequential(*(
                ([nn.BatchNorm1d(input_dim)] if self.use_bn else [])
                +
                [nn.Linear(input_dim, self.spatial_emb_dim),
                 nn.ReLU(inplace=True),
                 nn.Dropout(self.spatial_dropout)
                 ]
        ))
        # self.spatial_rels_bilstm = AlternatingHighwayLSTM(
        #     input_size=self.spatial_emb_dim,
        #     hidden_size=self.spatial_emb_dim,
        #     num_layers=1,
        #     recurrent_dropout_probability=self.recurrent_spatial_dropout)
        self.spatial_rel_ctx_bilstm = nn.LSTM(
            input_size=self.spatial_emb_dim,
            hidden_size=self.spatial_rnn_emb_dim,
            num_layers=1,
            # dropout=self.spatial_rnn_dropout,  # dropout requires a number of layers greater than 1, since it's not added after the last one
            bidirectional=True)

    @property
    def output_dim(self):
        return 2 * self.spatial_rnn_emb_dim  # 2 because of BiLSTM

    def forward(self, x, **kwargs):
        # TODO docs
        masks, unique_im_ids, hoi_im_ids, sub_inds, obj_inds = x
        with torch.set_grad_enabled(self.training):
            spatial_context = self._forward(masks, unique_im_ids, hoi_im_ids, sub_inds, obj_inds)
            return spatial_context

    def _forward(self, masks, unique_im_ids, hoi_im_ids, sub_inds, obj_inds):
        # TODO docs
        # Every input is a Tensor
        masks = masks.view([masks.shape[0], -1])
        ho_pair_masks = torch.cat([masks[sub_inds, :], masks[obj_inds, :]], dim=1)
        spatial_rels_feats = self.spatial_rels_fc(ho_pair_masks)
        spatial_ctx = compute_context(self.spatial_rel_ctx_bilstm, spatial_rels_feats, unique_im_ids, hoi_im_ids)
        return spatial_ctx


class ObjectContext(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()

        # FIXME params
        self.use_bn = False  # Since the batches are small due to memory constraint, BN is not suitable. TODO Maybe switch to GN?
        self.obj_fc_dim = 1024
        self.obj_rnn_emb_dim = 1024
        self.obj_rnn_dropout = 0.1

        self.obj_emb_fc = nn.Sequential(*[
            nn.Linear(input_dim, self.obj_fc_dim),
            nn.ReLU(inplace=True),
        ])
        self.obj_ctx_bilstm = nn.LSTM(
            input_size=self.obj_fc_dim,
            hidden_size=self.obj_rnn_emb_dim,
            num_layers=1,
            # dropout=self.obj_rnn_dropout,  # dropout requires a number of layers greater than 1, since it's not added after the last one
            bidirectional=True)

    def forward(self, x, **kwargs):
        # TODO docs
        boxes_ext, box_feats, spatial_ctx, unique_im_ids, box_im_ids = x
        with torch.set_grad_enabled(self.training):
            object_context, object_embeddings = self._forward(boxes_ext, box_feats, spatial_ctx, unique_im_ids, box_im_ids)
            return object_context, object_embeddings

    def _forward(self, boxes_ext, box_feats, spatial_ctx, unique_im_ids, box_im_ids):
        # TODO docs
        # Every input is a Tensor
        spatial_ctx_rep = torch.cat([spatial_ctx[i, :].expand((box_im_ids == im_id).sum(), -1) for i, im_id in enumerate(unique_im_ids)], dim=0)
        obj_embs = self.obj_emb_fc(torch.cat([box_feats, boxes_ext[:, 5:], spatial_ctx_rep], dim=1))
        obj_ctx = compute_context(self.obj_ctx_bilstm, obj_embs, unique_im_ids, box_im_ids)
        return obj_ctx, obj_embs

    @property
    def output_ctx_dim(self):
        return 2 * self.obj_rnn_emb_dim  # 2 because of BiLSTM

    @property
    def output_feat_dim(self):
        return self.obj_fc_dim


def compute_context(lstm, feats, im_ids, input_im_ids):
    # If I = #images, this is I x [N_i x feat_vec_dim], where N_i = #elements in image i
    feats_per_img = [feats[input_im_ids == im_id, :] for im_id in im_ids]

    feats_seq = nn.utils.rnn.pad_sequence(feats_per_img)  # this is max(N_i) x I x D
    context_feat_seq = lstm(feats_seq)[0]  # output is max(N_i) x I x 2 * hidden_state_dim
    # spatial_rel_ctx, _ = nn.utils.rnn.pad_packed_sequence(spatial_rel_ctx, batch_first=True)  # shouldn't be needed

    context_feats = context_feat_seq.mean(dim=0)  # this is I x whatever
    assert context_feats.shape[0] == len(im_ids)
    return context_feats
