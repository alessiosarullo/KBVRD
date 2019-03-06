import torch.nn as nn
import torch
import numpy as np
from lib.dataset.hicodet import HicoDetInstanceSplit
from lib.dataset.utils import Example


class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, dataset: HicoDetInstanceSplit, eps=1e-3):
        super(FrequencyBias, self).__init__()

        counts = self.get_counts(dataset)
        pred_dist = np.log(counts / np.maximum(1, np.sum(counts, axis=2, keepdims=True)) + eps)
        pred_dist = torch.tensor(pred_dist, dtype=torch.float32).view(-1, pred_dist.shape[2])

        self.num_objs = dataset.num_object_classes
        self.obj_baseline = nn.Embedding(pred_dist.shape[0], pred_dist.shape[1])
        self.obj_baseline.weight.data = pred_dist
        assert pred_dist.shape[0] == self.num_objs ** 2

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2] 
        :return: 
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def forward(self, obj_cands0, obj_cands1):
        """
        Here 151 = #objects and 51 = #predicates.
        :param obj_cands0: [batch_size, 151] prob distibution over cands.
        :param obj_cands1: [batch_size, 151] prob distibution over cands.
        :return: [batch_size, #predicates] array, which contains potentials for
        each possibility
        """
        # [batch_size, 151, 151] repr of the joint distribution
        joint_cands = obj_cands0[:, :, None] * obj_cands1[:, None]

        # [151, 151, 51] of targets per.
        baseline = joint_cands.view(joint_cands.size(0), -1) @ self.obj_baseline.weight

        return baseline

    @staticmethod
    def get_counts(train_data: HicoDetInstanceSplit):
        counts = np.zeros((train_data.num_object_classes, train_data.num_object_classes, train_data.num_predicates), dtype=np.int64)
        for i in range(len(train_data)):
            ex = train_data.get_entry(i, read_img=False)  # type: Example
            gt_hois = ex.gt_hois
            ho_pairs = ex.gt_obj_classes[gt_hois[:, [0, 2]]]
            for (h, o), pred in zip(ho_pairs, gt_hois[:, 1]):
                counts[h, o, pred] += 1
        return counts
