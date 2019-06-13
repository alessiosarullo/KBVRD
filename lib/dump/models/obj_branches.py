import numpy as np
import torch
import torch.nn as nn

from lib.dataset.hicodet.hicodet_split import HicoDetSplitBuilder, HicoDetSplit


class SpatialContext(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()

        self.use_bn = False
        self.hidden_spatial_repr_dim = 128
        self.spatial_rnn_repr_dim = 128
        self.dropout_rate = 0.1
        self.__dict__.update({k: v for k, v in kwargs.items() if k in self.__dict__.keys() and v is not None})

        self.spatial_rels_fc = nn.Sequential(*(
                [nn.Linear(input_dim, self.hidden_spatial_repr_dim),
                 nn.ReLU(inplace=True)]
                +
                ([nn.BatchNorm1d(self.hidden_spatial_repr_dim)] if self.use_bn else [])
        ))
        self.spatial_rel_ctx_bilstm = nn.LSTM(
            input_size=self.hidden_spatial_repr_dim,
            hidden_size=self.spatial_rnn_repr_dim,
            num_layers=1,
            bidirectional=True,
            # dropout=self.dropout_rate,
        )
        # self.spatial_rel_ctx_bilstm = AlternatingHighwayLSTM(input_size=self.hidden_spatial_repr_dim,
        #                                                      hidden_size=self.spatial_rnn_repr_dim,
        #                                                      num_layers=2,
        #                                                      recurrent_dropout_probability=self.dropout_rate)

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
            return spatial_context, rec_repr, spatial_rels_feats


class ObjectContext(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()

        # FIXME params
        # FIXME? Since batches are fairly small due to memory constraint, BN might not be suitable. Maybe switch to GN?
        self.use_bn = False
        self.obj_fc_dim = 512
        self.obj_rnn_emb_dim = self.obj_fc_dim // 2
        self.__dict__.update({k: v for k, v in kwargs.items() if k in self.__dict__.keys() and v is not None})

        self.obj_emb_fc = nn.Sequential(*[
            nn.Linear(input_dim, self.obj_fc_dim),
            nn.ReLU(inplace=True),
        ])
        self.obj_ctx_bilstm = nn.LSTM(
            input_size=self.obj_fc_dim,
            hidden_size=self.obj_rnn_emb_dim,
            num_layers=1,  # if you want to use dropout the number of layers has to be greater than 1, since it's not added after the last one
            bidirectional=True)

    def forward(self, boxes_ext, box_feats, unique_im_ids, box_im_ids, spatial_ctx=None, **kwargs):
        # TODO docs
        with torch.set_grad_enabled(self.training):
            if spatial_ctx is not None:
                spatial_ctx_rep = torch.cat([spatial_ctx[i, :].expand((box_im_ids == imid).sum(), -1) for i, imid in enumerate(unique_im_ids)], dim=0)
                object_embeddings = self.obj_emb_fc(torch.cat([box_feats, boxes_ext[:, 5:], spatial_ctx_rep], dim=1))
            else:
                object_embeddings = self.obj_emb_fc(torch.cat([box_feats, boxes_ext[:, 5:]], dim=1))
            object_context, rec_repr = compute_context(self.obj_ctx_bilstm, object_embeddings, unique_im_ids, box_im_ids)
            return object_context, rec_repr, object_embeddings

    @property
    def ctx_dim(self):
        return 2 * self.obj_rnn_emb_dim  # 2 because of BiLSTM

    @property
    def repr_dim(self):
        return self.obj_fc_dim


class SimpleObjBranch(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        self.obj_fc_dim = 512
        self.__dict__.update({k: v for k, v in kwargs.items() if k in self.__dict__.keys() and v is not None})

        self.obj_repr_fc = nn.Sequential(*[
            nn.Linear(input_dim, self.obj_fc_dim),
            nn.ReLU(inplace=True),
        ])

    def forward(self, boxes_ext, box_feats, **kwargs):
        with torch.set_grad_enabled(self.training):
            object_repr = self.obj_repr_fc(torch.cat([box_feats, boxes_ext[:, 5:]], dim=1))
            return object_repr

    @property
    def ctx_dim(self):
        raise NotImplementedError()

    @property
    def output_dim(self):
        return self.obj_fc_dim


class EmbObjBranch(nn.Module):
    def __init__(self, dataset: HicoDetSplit, vis_dim, **kwargs):
        super().__init__()
        self.obj_fc_dim = 512
        self.__dict__.update({k: v for k, v in kwargs.items() if k in self.__dict__.keys() and v is not None})

        entity_embs = np.load('cache/rotate/entity_embedding.npy')  # FIXME path
        with open('cache/rotate/entities.dict', 'r') as f:
            ecl_idx, entity_classes = zip(*[l.strip().split('\t') for l in f.readlines()])  # the index is loaded just for assertion check.
            ecl_idx = [int(x) for x in ecl_idx]
            assert np.all(np.arange(len(ecl_idx)) == np.array(ecl_idx))
            entity_inv_index = {e: i for i, e in enumerate(entity_classes)}
        obj_inds = np.array([entity_inv_index[o] for o in dataset.objects])
        obj_embs = entity_embs[obj_inds]

        self.obj_embs = nn.Parameter(torch.from_numpy(obj_embs), requires_grad=False)

        self.vis_repr_fc = nn.Sequential(*[
            nn.Linear(vis_dim + dataset.num_object_classes, self.obj_fc_dim),
            nn.ReLU(inplace=True),
        ])

        self.emb_repr_fc = nn.Sequential(*[
            nn.Linear(self.obj_embs.shape[1] + dataset.num_object_classes, self.obj_fc_dim),
            nn.ReLU(inplace=True),
        ])

        self.obj_repr_fc = nn.Sequential(*[
            nn.Linear(self.obj_fc_dim * 2, self.obj_fc_dim),
            nn.ReLU(inplace=True),
        ])

    def forward(self, boxes_ext, box_feats, unique_im_ids, box_im_ids, **kwargs):
        with torch.set_grad_enabled(self.training):
            vis_repr = self.vis_repr_fc(torch.cat([box_feats, boxes_ext[:, 5:]], dim=1))
            emb_repr = self.emb_repr_fc(torch.cat([boxes_ext[:, 5:].detach() @ self.obj_embs, boxes_ext[:, 5:]], dim=1))
            object_repr = self.obj_repr_fc(torch.cat([vis_repr, emb_repr], dim=1))
            return object_repr

    @property
    def ctx_dim(self):
        raise NotImplementedError()

    @property
    def output_dim(self):
        return self.obj_fc_dim


def compute_context(lstm, feats, im_ids, input_im_ids):
    # If I = #images, this is I x [N_i x D], where N_i = #elements in image i
    feats_per_img = [feats[input_im_ids == im_id, :] for im_id in im_ids]
    num_examples_per_img = [int((input_im_ids == im_id).sum().detach().cpu().item()) for im_id in im_ids]

    feats_seq = nn.utils.rnn.pad_sequence(feats_per_img)  # this is max(N_i) x I x D
    recurrent_repr_seq = lstm(feats_seq)[0]  # output is max(N_i) x I x H
    assert len(num_examples_per_img) == recurrent_repr_seq.shape[1]
    rec_repr_per_img = [recurrent_repr_seq[:num_ex, i, :] for i, num_ex in enumerate(num_examples_per_img)]
    rec_repr = torch.cat(rec_repr_per_img, dim=0)
    assert rec_repr.shape[0] == feats.shape[0] and rec_repr.shape[1] == lstm.hidden_size * 2, (rec_repr.shape, feats.shape[0], lstm.hidden_size)

    num_examples_per_img_torch = torch.from_numpy(np.array(num_examples_per_img)).to(recurrent_repr_seq).view(-1, 1)
    context_feats = recurrent_repr_seq.sum(dim=0) / num_examples_per_img_torch  # this is I x H
    assert context_feats.shape[0] == len(im_ids)
    context_feats = torch.cat([context_feats[i, :].expand(num_ex, -1) for i, num_ex in enumerate(num_examples_per_img)], dim=0)  # N x H
    return context_feats, rec_repr
