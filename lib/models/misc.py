import numpy as np
import torch
from torch.nn import functional

from config import cfg
from lib.dataset.hicodet.hicodet_split import HicoDetSplit


def bce_loss(logits, labels, reduce=True):
    if cfg.fl_gamma != 0:  # Focal loss
        gamma = cfg.fl_gamma
        s = logits
        t = labels
        m = s.clamp(min=0)  # m = max(s, 0)
        x = (-s.abs()).exp()
        z = ((s >= 0) == t.byte()).float()
        loss_mat = (1 + x).pow(-gamma) * (m - s * t + x * (gamma * z).exp() * (1 + x).log())
        if reduce:
            loss = loss_mat.mean()
        else:
            loss = loss_mat
    else:  # standard BCE loss
        loss = functional.binary_cross_entropy_with_logits(logits, labels, reduction='elementwise_mean' if reduce else 'none')
    if reduce and not cfg.meanc:
        loss *= logits.shape[1]
    return loss


def LIS(x, w=None, k=None, T=None):  # defaults are as in the paper
    if T is None:
        if w is None and k is None:
            w, k, T = 10, 12, 8.4
        else:
            assert w is not None and k is not None
            # This is basically what it is: a normalisation constant for when x=1.
            T = 1 + np.exp(k - w).item()
    assert w is not None and k is not None and T is not None
    return T * torch.sigmoid(w * x - k)


def interactions_to_actions(hois, hico):
    i_to_a_mat = np.zeros((hico.num_interactions, hico.num_actions))
    i_to_a_mat[np.arange(hico.num_interactions), hico.interactions[:, 0]] = 1
    i_to_a_mat = torch.from_numpy(i_to_a_mat).to(hois)
    actions = (hois @ i_to_a_mat).clamp(max=1)
    return actions


def interactions_to_objects(hois, hico):
    i_to_o_mat = np.zeros((hico.num_interactions, hico.num_object_classes))
    i_to_o_mat[np.arange(hico.num_interactions), hico.interactions[:, 1]] = 1
    i_to_o_mat = torch.from_numpy(i_to_o_mat).to(hois)
    objects = (hois @ i_to_o_mat).clamp(max=1)
    return objects


def interactions_to_mat(hois, hico):
    hois_np = hois.detach().cpu().numpy()
    all_hois = np.stack(np.where(hois_np > 0), axis=1)
    all_interactions = np.concatenate([all_hois[:, :1], hico.interactions[all_hois[:, 1], :]], axis=1)
    inter_mat = np.zeros((hois.shape[0], hico.num_object_classes, hico.num_actions))
    inter_mat[all_interactions[:, 0], all_interactions[:, 2], all_interactions[:, 1]] = 1
    # TODO inter_mat[:, :, 0] = 0
    inter_mat = torch.from_numpy(inter_mat).to(hois)
    return inter_mat


def get_hoi_adjacency_matrix(dataset, isolate_null=None):
    if isolate_null is None:
        isolate_null = cfg.iso_null
    interactions = dataset.full_dataset.interactions
    inter_obj_adj = np.zeros((dataset.full_dataset.num_interactions, dataset.full_dataset.num_object_classes))
    inter_obj_adj[np.arange(interactions.shape[0]), interactions[:, 0]] = 1

    inter_act_adj = np.zeros((dataset.full_dataset.num_interactions, dataset.full_dataset.num_actions))
    inter_act_adj[np.arange(interactions.shape[0]), interactions[:, 1]] = 1

    adj = inter_obj_adj @ inter_obj_adj.T + inter_act_adj @ inter_act_adj.T
    adj = torch.from_numpy(adj).clamp(max=1).byte()

    if isolate_null:
        null_hois = np.flatnonzero(np.any(inter_act_adj[:, 1:], axis=1))
        adj_pos = adj
        adj_neg = ~adj  # Note: don't move down! I'm not sure it would still work.

        adj_pos[null_hois, :] = 0
        adj_pos[:, null_hois] = 0
        adj_neg[null_hois, :] = 0
        adj_neg[:, null_hois] = 0
        return adj_pos, adj_neg
    else:
        return adj


def get_noun_verb_adj_mat(dataset: HicoDetSplit, isolate_null=None):
    if isolate_null is None:
        isolate_null = cfg.iso_null
    noun_verb_links = torch.from_numpy((dataset.full_dataset.op_pair_to_interaction >= 0).astype(np.float32))
    if isolate_null:
        noun_verb_links[:, 0] = 0
    return noun_verb_links
