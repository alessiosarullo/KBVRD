import torch
import torch.nn as nn


class SpatialContext(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()

        self.use_bn = False
        self.hidden_spatial_repr_dim = 128
        self.spatial_rnn_repr_dim = 128
        self.__dict__.update({k: v for k, v in kwargs.items() if k in self.__dict__.keys() and v is not None})

        self.spatial_rels_fc = nn.Sequential(*(
                [nn.Linear(input_dim, self.hidden_spatial_repr_dim),
                 nn.ReLU(inplace=True)]
                +
                ([nn.BatchNorm1d(self.hidden_spatial_repr_dim)] if self.use_bn else [])
        ))
        # self.spatial_rels_bilstm = AlternatingHighwayLSTM(
        #     input_size=self.spatial_emb_dim,
        #     hidden_size=self.spatial_emb_dim,
        #     num_layers=1,
        #     recurrent_dropout_probability=self.recurrent_spatial_dropout)
        self.spatial_rel_ctx_bilstm = nn.LSTM(
            input_size=self.hidden_spatial_repr_dim,
            hidden_size=self.spatial_rnn_repr_dim,
            num_layers=1,  # if you want to use dropout the number of layers has to be greater than 1, since it's not added after the last one
            bidirectional=True)

    @property
    def context_dim(self):
        return 2 * self.spatial_rnn_repr_dim  # 2 because of BiLSTM

    @property
    def repr_dim(self):
        return 2 * self.spatial_rnn_repr_dim

    def forward(self, masks, unique_im_ids, hoi_infos, **kwargs):
        # TODO docs
        with torch.set_grad_enabled(self.training):
            hoi_im_ids = hoi_infos[:, 0]
            sub_inds = hoi_infos[:, 1]
            obj_inds = hoi_infos[:, 2]

            masks = masks.view([masks.shape[0], -1])
            ho_pair_masks = torch.cat([masks[sub_inds, :], masks[obj_inds, :]], dim=1)
            spatial_rels_feats = self.spatial_rels_fc(ho_pair_masks)
            spatial_context, rec_repr = compute_context(self.spatial_rel_ctx_bilstm, spatial_rels_feats, unique_im_ids, hoi_im_ids)
            return spatial_context, rec_repr


class ObjectContext(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()

        # FIXME params
        # FIXME? Since batches are fairly small due to memory constraint, BN might not be suitable. Maybe switch to GN?
        self.use_bn = False
        self.obj_fc_dim = 1024
        self.obj_rnn_emb_dim = 1024

        self.obj_emb_fc = nn.Sequential(*[
            nn.Linear(input_dim, self.obj_fc_dim),
            nn.ReLU(inplace=True),
        ])
        self.obj_ctx_bilstm = nn.LSTM(
            input_size=self.obj_fc_dim,
            hidden_size=self.obj_rnn_emb_dim,
            num_layers=1,  # if you want to use dropout the number of layers has to be greater than 1, since it's not added after the last one
            bidirectional=True)

    def forward(self, boxes_ext, box_feats, spatial_ctx, unique_im_ids, box_im_ids, **kwargs):
        # TODO docs
        with torch.set_grad_enabled(self.training):
            spatial_ctx_rep = torch.cat([spatial_ctx[i, :].expand((box_im_ids == im_id).sum(), -1) for i, im_id in enumerate(unique_im_ids)], dim=0)
            object_embeddings = self.obj_emb_fc(torch.cat([box_feats, boxes_ext[:, 5:], spatial_ctx_rep], dim=1))
            object_context, rec_repr = compute_context(self.obj_ctx_bilstm, object_embeddings, unique_im_ids, box_im_ids)
            # FIXME return rec_repr instead of pre recurrent representation
            return object_context, object_embeddings

    @property
    def output_ctx_dim(self):
        return 2 * self.obj_rnn_emb_dim  # 2 because of BiLSTM

    @property
    def output_repr_dim(self):
        return self.obj_fc_dim


def compute_context(lstm, feats, im_ids, input_im_ids):
    # If I = #images, this is I x [N_i x feat_vec_dim], where N_i = #elements in image i
    feats_per_img = [feats[input_im_ids == im_id, :] for im_id in im_ids]
    num_examples_per_img = [int((input_im_ids == im_id).sum().detach().cpu().item()) for im_id in im_ids]

    feats_seq = nn.utils.rnn.pad_sequence(feats_per_img)  # this is max(N_i) x I x D
    recurrent_repr_seq = lstm(feats_seq)[0]  # output is max(N_i) x I x 2 * hidden_state_dim
    assert len(num_examples_per_img) == recurrent_repr_seq.shape[1]
    rec_repr = torch.stack([recurrent_repr_seq[n, i, :] for i, num_ex in enumerate(num_examples_per_img) for n in range(num_ex)], dim=0)
    assert rec_repr.shape[0] == feats.shape[0] and rec_repr.shape[1] == lstm.hidden_size * 2, (rec_repr.shape, feats.shape[0], lstm.hidden_size)

    # FIXME! This might not work well with padding (i.e., max(N_i)). The mean should only be computed across objects actually in the image.
    context_feats = recurrent_repr_seq.mean(dim=0)  # this is I x whatever
    assert context_feats.shape[0] == len(im_ids)
    return context_feats, rec_repr
