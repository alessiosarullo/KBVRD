import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence

from config import cfg
from lib.models.nmotifs.word_vectors import obj_edge_vectors
from .decoder_rnn import DecoderRNN
from .pytorch_misc import arange, enumerate_by_image, transpose_packed_sequence_inds
from ..highway_lstm_cuda.alternating_highway_lstm import AlternatingHighwayLSTM


class LinearizedContext(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, classes, visual_feat_dim, **kwargs):
        super(LinearizedContext, self).__init__()

        assert not (set(kwargs.keys()) - set(self.__dict__.keys()))

        # Defaults used in NeuralMotifs
        pos_emb_dim = 128
        self.word_emb_dim = 200
        self.obj_ctx_num_layers = 2
        self.edge_ctx_num_layers = 4
        self.order = 'leftright'
        self.rnn_hidden_dim = 256
        self.dropout_rate = 0.1

        self.__dict__.update(kwargs)
        assert self.order in ('size', 'confidence', 'random', 'leftright')
        assert self.obj_ctx_num_layers > 0 and self.edge_ctx_num_layers > 0

        self.classes = classes

        # EMBEDDINGS
        embed_vecs = obj_edge_vectors(self.classes, wv_dim=self.word_emb_dim)
        self.obj_embed = nn.Embedding(self.num_classes, self.word_emb_dim)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(self.num_classes, self.word_emb_dim)
        self.obj_embed2.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(nn.BatchNorm1d(4, momentum=cfg.model.bn_momentum / 10.0),
                                       nn.Linear(4, pos_emb_dim),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1),
                                       )

        self.obj_ctx_rnn = AlternatingHighwayLSTM(input_size=visual_feat_dim + self.word_emb_dim + pos_emb_dim,
                                                  hidden_size=self.rnn_hidden_dim,
                                                  num_layers=self.obj_ctx_num_layers,
                                                  recurrent_dropout_probability=self.dropout_rate)

        self.decoder_rnn = DecoderRNN(self.classes, embed_dim=self.word_emb_dim,
                                      inputs_dim=self.rnn_hidden_dim,
                                      hidden_dim=self.rnn_hidden_dim,
                                      recurrent_dropout_probability=self.dropout_rate)

        self.edge_ctx_rnn = AlternatingHighwayLSTM(input_size=self.word_emb_dim + self.rnn_hidden_dim,
                                                   hidden_size=self.rnn_hidden_dim,
                                                   num_layers=self.edge_ctx_num_layers,
                                                   recurrent_dropout_probability=self.dropout_rate)

    @property
    def edge_ctx_dim(self):
        return self.rnn_hidden_dim

    @property
    def num_classes(self):
        return len(self.classes)

    def forward(self, obj_vis_feats, boxes_ext, box_labels):
        """
        Forward pass through the object and edge context
        :param obj_priors:
        :param obj_vis_feats:
        :param im_inds:
        :param boxes:
        :return:
        """

        im_inds = boxes_ext[:, 0].long()
        box_priors = boxes_ext[:, 1:5]
        obj_logits = boxes_ext[:, 5:].detach()

        obj_embed = F.softmax(obj_logits, dim=1) @ self.obj_embed.weight
        pos_embed = self.pos_embed(center_size(box_priors))
        obj_pre_rep = torch.cat((obj_vis_feats, obj_embed, pos_embed), dim=1)

        obj_dists, obj_preds, obj_ctx = self.obj_ctx(obj_feats=obj_pre_rep,
                                                     obj_logits=obj_logits,
                                                     im_inds=im_inds,
                                                     box_priors=box_priors,
                                                     box_labels=box_labels,
                                                     )

        edge_ctx = self.edge_ctx(obj_ctx,
                                 obj_dists=obj_dists.detach(),
                                 im_inds=im_inds,
                                 obj_preds=obj_preds,
                                 box_priors=box_priors,
                                 )

        return obj_dists, obj_preds, edge_ctx

    def obj_ctx(self, obj_feats, obj_logits, im_inds, box_priors, box_labels):
        """
        Object context and object classification.
        :param obj_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_logits: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :param obj_labels: [num_obj] the GT labels of the image
        :param boxes: [num_obj, 4] boxes. We'll use this for NMS
        :return: obj_dists: [num_obj, #classes] new probability distribution.
                 obj_preds: argmax of that distribution.
                 obj_final_ctx: [num_obj, #feats] For later!
        """
        # Sort by the confidence of the maximum detection.
        confidence = F.softmax(obj_logits, dim=1).data[:, 1:].max(1)[0]
        perm, inv_perm, ls_transposed = sort_rois(self.order, im_inds.data, box_priors, confidence)

        # Pass object features, sorted by score, into the encoder LSTM
        obj_inp_rep = obj_feats[perm].contiguous()
        input_packed = PackedSequence(obj_inp_rep, ls_transposed)
        encoder_rep = self.obj_ctx_rnn(input_packed)[0]

        # Decode in order
        decoder_inp = PackedSequence(encoder_rep, ls_transposed)
        obj_logits, obj_preds = self.decoder_rnn(decoder_inp, labels=box_labels)
        obj_preds = obj_preds[inv_perm]
        obj_logits = obj_logits[inv_perm]
        encoder_rep = encoder_rep[inv_perm]

        return obj_logits, obj_preds, encoder_rep

    def edge_ctx(self, obj_ctx_feats, obj_dists, im_inds, obj_preds, box_priors):
        """
        Object context and object classification.
        :param obj_ctx_feats: [num_obj, img_dim + object embedding0 dim]
        :param obj_dists: [num_obj, #classes]
        :param im_inds: [num_obj] the indices of the images
        :return: edge_ctx: [num_obj, #feats] For later!
        """

        # Only use hard embeddings
        obj_embed2 = self.obj_embed2(obj_preds)
        # obj_embed3 = F.softmax(obj_dists, dim=1) @ self.obj_embed3.weight
        inp_feats = torch.cat((obj_embed2, obj_ctx_feats), 1)

        # Sort by the confidence of the maximum detection.
        confidence = F.softmax(obj_dists, dim=1).data.view(-1)[
            obj_preds.data + arange(obj_preds.data) * self.num_classes]
        perm, inv_perm, ls_transposed = sort_rois(self.order, im_inds.data, box_priors, confidence)

        edge_input_packed = PackedSequence(inp_feats[perm], ls_transposed)
        edge_reps = self.edge_ctx_rnn(edge_input_packed)[0]

        # now we're good! unperm
        edge_ctx = edge_reps[inv_perm]
        return edge_ctx


def sort_rois(order, batch_idx, box_priors=None, confidence=None):
    """
    :param batch_idx: tensor with what index we're on
    :param confidence: tensor with confidences between [0,1)
    :param boxes: tensor with (x1, y1, x2, y2)
    :return: Permutation, inverse permutation, and the lengths transposed (same as _sort_by_score)
    """

    if order == 'size':
        cxcywh = center_size(box_priors)
        sizes = cxcywh[:, 2] * cxcywh[:, 3]
        assert sizes.min() > 0.0
        scores = sizes / (sizes.max() + 1)
    elif order == 'confidence':
        assert confidence is not None
        scores = confidence
    elif order == 'random':
        scores = torch.FloatTensor(np.random.rand(batch_idx.size(0))).cuda(batch_idx.get_device())
    elif order == 'leftright':
        cxcywh = center_size(box_priors)
        centers = cxcywh[:, 0]
        scores = centers / (centers.max() + 1)
    else:
        raise ValueError("invalid mode {}".format(order))
    return _sort_by_score(batch_idx, scores)


def _sort_by_score(im_inds, scores):
    """
    We'll sort everything scorewise from Hi->low, BUT we need to keep images together
    and sort LSTM from l
    :param im_inds: Which im we're on
    :param scores: Goodness ranging between [0, 1]. Higher numbers come FIRST
    :return: Permutation to put everything in the right order for the LSTM
             Inverse permutation
             Lengths for the TxB packed sequence.
    """
    num_im = im_inds.max() + 1
    rois_per_image = scores.new_empty(num_im)
    lengths = []
    for i, s, e in enumerate_by_image(im_inds):
        rois_per_image[i] = 2 * (s - e) * num_im + i
        lengths.append(e - s)
    lengths = sorted(lengths, reverse=True)
    inds, ls_transposed = transpose_packed_sequence_inds(lengths)  # move it to TxB form
    inds = torch.LongTensor(inds).cuda(im_inds.get_device())

    # ~~~~~~~~~~~~~~~~
    # HACKY CODE ALERT!!!
    # we're sorting by confidence which is in the range (0,1), but more importantly by longest
    # img....
    # ~~~~~~~~~~~~~~~~
    roi_order = scores - 2 * rois_per_image[im_inds]
    _, perm = torch.sort(roi_order, 0, descending=True)
    perm = perm[inds]
    _, inv_perm = torch.sort(perm)

    return perm, inv_perm, torch.from_numpy(np.array(ls_transposed))


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    wh = boxes[:, 2:] - boxes[:, :2] + 1.0

    if isinstance(boxes, np.ndarray):
        return np.column_stack((boxes[:, :2] + 0.5 * wh, wh))
    return torch.cat((boxes[:, :2] + 0.5 * wh, wh), 1)
