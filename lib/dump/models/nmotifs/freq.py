import torch.nn as nn
import torch
import numpy as np


class FrequencyLogProbs(nn.Module):
    def __init__(self, counts: np.ndarray, eps=1e-3, freeze=False):
        super().__init__()

        log_probs = np.log(counts / np.maximum(1, np.sum(counts, axis=1, keepdims=True)) + eps)
        self.counts = counts
        self.log_probs_embs = torch.nn.Embedding.from_pretrained(torch.from_numpy(log_probs).float(), freeze=freeze)

        # # Old style
        # log_probs = torch.tensor(log_probs, dtype=torch.float32)
        # self.log_probs_embs = nn.Embedding(log_probs.shape[0], log_probs.shape[1])
        # self.log_probs_embs.weight.data = log_probs

    def forward(self, labels):
        """
        :param labels: array of object predicted labels (no probability distribution)
        :return:
        """
        return self.log_probs_embs(labels)
