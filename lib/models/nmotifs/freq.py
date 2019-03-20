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

        self.counts = self.get_counts(dataset)
        pred_dist = np.log(self.counts / np.maximum(1, np.sum(self.counts, axis=1, keepdims=True)) + eps)
        pred_dist = torch.tensor(pred_dist, dtype=torch.float32)

        assert pred_dist.shape[0] == dataset.num_object_classes
        self.obj_baseline = nn.Embedding(pred_dist.shape[0], pred_dist.shape[1])
        self.obj_baseline.weight.data = pred_dist

    def index_with_labels(self, obj_labels):
        return self.obj_baseline(obj_labels)  # obj_labels is an array of object predicted labels (no probability distribution)

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
        counts = np.zeros((train_data.num_object_classes, train_data.num_predicates), dtype=np.int64)
        for i in range(len(train_data)):
            ex = train_data.get_entry(i, read_img=False, ignore_precomputed=True)  # type: Example
            gt_hois = ex.gt_hois
            objs = ex.gt_obj_classes[gt_hois[:, 2]]
            assert np.all(ex.gt_obj_classes[gt_hois[:, 0]] == train_data.human_class)
            for o, pred in zip(objs, gt_hois[:, 1]):
                counts[o, pred] += 1
        return counts
