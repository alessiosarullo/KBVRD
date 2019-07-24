import time
from collections import Counter
from typing import Dict

import numpy as np
import torch


class Timer:
    global_timer = None
    gpu_sync = False  # This is useful to time GPU operations, which is otherwise inaccurate due to asynchronous computations.

    def __init__(self):
        self.start_time = None
        self.last = None
        self.total_time = 0
        self.num_instances = 0
        self.sub_timers = {}  # type:Dict[str, Timer]

    @classmethod
    def get(cls, *args, get_only=False):
        if cls.global_timer is None:
            cls.global_timer = cls()
        timer = cls.global_timer
        for subt in args:
            if get_only:
                try:
                    timer = timer.sub_timers[subt]
                except KeyError:
                    raise ValueError('Unknown timer %s.' % subt)
            else:
                timer = timer.sub_timers.setdefault(subt, Timer())
        return timer  # type: Timer

    def tic(self):
        global_t = self.get()
        if global_t.start_time is None:
            global_t._tic()
        self._tic()

    def _tic(self):
        assert self.start_time is None
        self.start_time = time.perf_counter()

    def toc(self, synchronize=False):
        if synchronize and self.__class__.gpu_sync:
            torch.cuda.synchronize()
        self.last = time.perf_counter() - self.start_time
        self.total_time += self.last
        self.start_time = None
        self.num_instances += 1

    def progress(self):
        return time.perf_counter() - self.start_time

    def spent(self, average=True):
        return self.total_time / (self.num_instances if average else 1)

    def _get_lines(self, average):
        sep = ' ' * 4
        s = ['%s (x%d)' % (self.format(self.spent(average=average)), self.num_instances)]
        for k, v in self.sub_timers.items():
            try:
                sub_s = v._get_lines(average)
            except ZeroDivisionError:
                print(k)
                raise
            s.append('%s %s: %s' % (sep, k, sub_s[0]))
            s += ['%s %s' % (sep, ss) for ss in sub_s[1:]]
        return s

    def print(self, average=True):
        if self.total_time == 0 and self.start_time is not None:
            self.toc()
        print('Total time:', '\n'.join(self._get_lines(average=average)))

    @staticmethod
    def format(seconds):
        if seconds < 0.001:
            s, unit = '%.2f' % (seconds * 1000), 'ms'
        elif seconds < 0.01:
            s, unit = '%.1f' % (seconds * 1000), 'ms'
        elif seconds < 1:
            s, unit = '%.0f' % (seconds * 1000), 'ms'
        elif seconds < 10:
            s, unit = '%.1f' % seconds, 's'
        elif seconds < 60:
            s, unit = '%.0f' % seconds, 's'
        elif seconds < 600:
            s, unit = '%d:%02d' % divmod(seconds, 60), 'm'
        elif seconds < 3600:
            s, unit = '%.0f' % (seconds / 60), 'm'
        else:
            s, unit = '%d:%02d' % divmod(seconds / 60, 60), 'h'
        return '%s%s' % (s, unit)


def sort_and_filter(metrics, gt_labels, all_classes, sort=False, keep_inds=None):
    gt_labels_hist = Counter(gt_labels)
    for c in all_classes:
        gt_labels_hist.setdefault(c, 0)

    if keep_inds:
        del_inds = set(gt_labels_hist.keys()) - set(keep_inds)
        for k in del_inds:
            del gt_labels_hist[k]

    if sort:
        class_inds = [p for p, num in gt_labels_hist.most_common()]
    else:
        class_inds = sorted(gt_labels_hist.keys())

    metrics = {k: v[class_inds] if v.size > 1 else v for k, v in metrics.items()}
    return gt_labels_hist, metrics, class_inds


class MetricFormatter:
    def __init__(self):
        super().__init__()

    def format_metric_and_gt_lines(self, gt_label_hist, metrics, class_inds, gt_str=None, print_out=True):
        num_gt = sum(gt_label_hist.values())

        pad = len(gt_str) if gt_str is not None else 0
        if metrics:
            pad = max(pad, max([len(k) for k in metrics.keys()]))

        p = 2
        lines = []
        for k, v in metrics.items():
            lines += [self.format_metric(k, v, pad, precision=p)]
        format_str = '%{}s %{}s [%s]'.format(pad + 1, p + 8)
        if gt_str is not None:
            lines += [format_str % ('%s:' % gt_str, 'IDs', ' '.join([('%{:d}d '.format(p + 5)) % i for i in class_inds]))]
            lines += [format_str % ('', '%', ' '.join([self._format_percentage(gt_label_hist[i] / num_gt, precision=p) for i in class_inds]))]

        if print_out:
            print('\n'.join(lines))
        return lines

    def format_metric(self, metric_name, data, metric_str_len=None, **kwargs):
        metric_str_len = metric_str_len or len(metric_name)
        per_class_str = ' @ [%s]' % ' '.join([self._format_percentage(x, **kwargs) for x in data]) if data.size > 1 else ''
        f_str = ('%{}s: %s%s'.format(metric_str_len)) % (metric_name, self._format_percentage(np.mean(data), **kwargs), per_class_str)
        return f_str

    @staticmethod
    def _format_percentage(value, precision=2):
        if -1 < value < 1:
            if value != 0:
                return ('% {}.{}f%%'.format(precision + 5, precision)) % (value * 100)
            else:
                return ('% {}.{}f%%'.format(precision + 5, 0)) % (value * 100)
        else:
            return ('% {}d%%'.format(precision + 5)) % (100 * np.sign(value))