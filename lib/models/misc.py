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


def get_noun_verb_adj_mat(dataset: HicoDetSplit, iso_null=None):
    noun_verb_links = torch.from_numpy((dataset.full_dataset.op_pair_to_interaction >= 0).astype(np.float32))
    if iso_null is None:
        iso_null = cfg.iso_null
    if iso_null:
        noun_verb_links[:, 0] = 0
    return noun_verb_links
