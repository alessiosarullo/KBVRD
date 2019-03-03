import os
import numpy as np

from config import cfg
import torch
from collections import deque
from typing import Dict
from tensorboardX import SummaryWriter


class SmoothedValue:
    """Track a series of values and provide access to smoothed values over a window or the global series average. """

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def append(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    def get_median(self):
        return np.median(self.deque)

    def get_average(self):
        return np.mean(self.deque)

    def get_global_average(self):
        return self.total / self.count


class TrainingStats:
    def __init__(self, num_batches, smoothing_window=20):
        # Output logging period in SGD iterations
        os.makedirs(cfg.program.tensorboard_dir)
        self.num_batches = num_batches
        self.history_window = smoothing_window

        self.tblogger = SummaryWriter(cfg.program.tensorboard_dir)
        self.tb_ignored_keys = ['iter']
        self.smoothed_losses = {}  # type: Dict[str, SmoothedValue]
        self.smoothed_metrics = {}  # type: Dict[str, SmoothedValue]
        self.smoothed_total_loss = SmoothedValue(self.history_window)
        self.values_to_watch = {}

    def update_stats(self, output_dict):
        assert sum([int('total' in k.lower()) for k in output_dict['losses'].keys()]) == 1
        for loss_name, loss in output_dict['losses'].items():
            if 'total' not in loss_name.lower():
                self.smoothed_losses.setdefault(loss_name, SmoothedValue(self.history_window)).append(loss.item())
            else:
                self.smoothed_total_loss.append(output_dict['losses'][loss_name].item())
        for metric_name, metric in output_dict.get('metrics', {}).items():
            self.smoothed_metrics.setdefault(metric_name, SmoothedValue(self.history_window)).append(metric.item())
        for name, value in output_dict.get('watch', {}).items():
            self.values_to_watch.setdefault(name, deque(maxlen=100)).append(value)  # FIXME magic constant

    def log_stats(self, curr_iter, lr):
        """Log the tracked statistics."""
        stats = {'Total loss': self.smoothed_total_loss.get_average(),
                 'LR': lr,
                 'Metrics': {k: v.get_median() for k, v in self.smoothed_metrics.items()},
                 'Watch': {k: torch.cat(tuple(v), dim=0) for k, v in self.values_to_watch.items()},
                 }
        for k, v in self.smoothed_losses.items():
            loss_name = k.replace('_', ' ').capitalize().replace('hoi', 'HOI').replace('Hoi', 'HOI')
            stats[loss_name] = v.get_average()
            print('%-20s %f' % (loss_name, stats[loss_name]))
        print('%-20s %f' % ('Total loss', stats['Total loss']))

        if self.tblogger is not None:
            self._tb_log_stats(stats, curr_iter)

    def _tb_log_stats(self, stats, curr_iter):
        """Log the tracked statistics to tensorboard"""
        for k in stats:
            if k not in self.tb_ignored_keys:
                v = stats[k]
                if isinstance(v, dict):
                    self._tb_log_stats(v, curr_iter)
                elif isinstance(v, (torch.Tensor, np.ndarray)):
                    self.tblogger.add_histogram(k, v, global_step=curr_iter)
                else:
                    self.tblogger.add_scalar(k, v, global_step=curr_iter)

    def close_tensorboard_logger(self):
        if self.tblogger is not None:
            self.tblogger.close()
