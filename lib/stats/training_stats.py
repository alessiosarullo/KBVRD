import os
import numpy as np
import shutil

from config import cfg
import torch
from collections import deque
from typing import Dict
from tensorboardX import SummaryWriter

from lib.dataset.utils import Splits
from lib.stats.utils import Timer
from torch.utils.data import DataLoader


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
    def __init__(self, split, data_loader: DataLoader, history_window=None):
        if history_window is None:
            history_window = cfg.program.log_interval

        self.split = split
        self.data_loader = data_loader
        self.history_window = history_window
        self.tb_ignored_keys = ['iter']

        tboard_dir = os.path.join(cfg.program.tensorboard_dir, self.split_str)
        try:
            shutil.rmtree(tboard_dir)
        except FileNotFoundError:
            pass
        os.makedirs(tboard_dir)
        self.tblogger = SummaryWriter(tboard_dir)
        self.smoothed_total_loss = SmoothedValue(self.history_window)
        self.smoothed_losses = {}  # type: Dict[str, SmoothedValue]
        self.smoothed_metrics = {}  # type: Dict[str, SmoothedValue]
        self.values_to_watch = {}  # type: Dict[str, SmoothedValue]
        self.histograms = {}

    @property
    def split_str(self):
        return self.split.value.capitalize()

    @property
    def epoch_str(self):
        return '%s epoch' % self.split_str

    def update_stats(self, output_dict):
        assert sum([int('total' in k.lower()) for k in output_dict['losses'].keys()]) == 1
        for loss_name, loss in output_dict['losses'].items():
            if 'total' not in loss_name.lower():
                self.smoothed_losses.setdefault(loss_name, SmoothedValue(self.history_window)).append(loss.item())
            else:
                self.smoothed_total_loss.append(output_dict['losses'][loss_name].item())
        for metric_name, metric in output_dict.get('metrics', {}).items():
            self.smoothed_metrics.setdefault(metric_name, SmoothedValue(self.history_window)).append(metric.item())
        for name, value in output_dict.get('hist', {}).items():
            self.histograms.setdefault(name, deque(maxlen=self.history_window)).append(value)
        for name, value in output_dict.get('watch', {}).items():
            self.values_to_watch.setdefault(name, SmoothedValue(self.history_window)).append(value.item())

    def log_stats(self, curr_iter, epoch, batch=None, verbose=False, **kwargs):
        """Log the tracked statistics."""
        Timer.get(self.epoch_str, 'Stats').tic()
        if verbose:
            self._print_times(epoch, batch=batch, curr_iter=curr_iter)

        stats = {'Total loss': self.smoothed_total_loss.get_average(),
                 'Metrics': {k: v.get_average() for k, v in self.smoothed_metrics.items()},
                 'Watch': {k: v.get_average() for k, v in self.values_to_watch.items()},
                 'Hist': {k: torch.cat(tuple(v), dim=0) for k, v in self.histograms.items()},
                 }
        for k, v in kwargs.items():
            stats[k] = v

        for k, v in self.smoothed_losses.items():
            loss_name = k.replace('_', ' ').capitalize().replace('hoi', 'HOI').replace('Hoi', 'HOI')
            stats[loss_name] = v.get_average()
            if verbose:
                print('%-20s %f' % (loss_name, stats[loss_name]))

        if verbose:
            print('%-20s %f' % ('Total loss', stats['Total loss']))
            if cfg.program.verbose:
                for k, v in stats['Hist'].items():
                    print('%30s: mean=% 6.4f, std=%6.4f' % (k, v.mean(), v.std()))
            print('-' * 10, flush=True)

        if self.tblogger is not None:
            self._tb_log_stats(stats, curr_iter)
        Timer.get(self.epoch_str, 'Stats').toc()

    def epoch_tic(self):
        Timer.get(self.epoch_str).tic()

    def epoch_toc(self):
        epoch_timer = Timer.get(self.epoch_str, get_only=True)
        epoch_timer.toc()
        print('Time for epoch:', Timer.format(epoch_timer.last))
        print('-' * 100, flush=True)

    def batch_tic(self):
        Timer.get(self.epoch_str, 'Batch').tic()

    def batch_toc(self):
        Timer.get(self.epoch_str, 'Batch').toc()

    def _print_times(self, epoch, **kwargs):
        num_batches = len(self.data_loader)
        time_per_batch = Timer.get(self.epoch_str, 'Batch', get_only=True).spent(average=True)
        time_to_load = Timer.get('GetBatch', get_only=True).spent(average=True)
        time_to_collate = Timer.get('Collate', get_only=True).spent(average=True)
        est_time_per_epoch = num_batches * (time_per_batch + time_to_load * self.data_loader.batch_size + time_to_collate)

        if self.split == Splits.VAL:
            header = '{:s} epoch {:2d}.'.format(self.split_str, epoch)
        else:
            try:
                batch = kwargs['batch']
                curr_iter = kwargs['curr_iter']
            except KeyError:
                raise
            header = '{:s} iter {:6d} (epoch {:2d}, batch {:5d}/{:5d}).'.format(self.split_str, curr_iter, epoch, batch, num_batches - 1)

        print(header, 'Avg: {:>5s}/batch, {:>5s}/load, {:>5s}/collate.'.format(Timer.format(time_per_batch),
                                                                               Timer.format(time_to_load),
                                                                               Timer.format(time_to_collate)),
              'Current epoch progress: {:>7s}/{:>7s} (estimated).'.format(Timer.format(Timer.get(self.epoch_str, get_only=True).progress()),
                                                                          Timer.format(est_time_per_epoch)))

    def _tb_log_stats(self, stats, curr_iter):
        """Log the tracked statistics to tensorboard"""
        for k in stats:
            if k not in self.tb_ignored_keys:
                v = stats[k]
                if isinstance(v, dict):
                    self._tb_log_stats(v, curr_iter)
                elif isinstance(v, (torch.Tensor, np.ndarray)):
                    if curr_iter > 0:
                        self.tblogger.add_histogram(k, v, global_step=curr_iter, bins='auto')
                else:
                    self.tblogger.add_scalar(k, v, global_step=curr_iter)

    def close_tensorboard_logger(self):
        if self.tblogger is not None:
            self.tblogger.close()
