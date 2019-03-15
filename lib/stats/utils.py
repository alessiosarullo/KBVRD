import time
from typing import Dict

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

    @classmethod
    def print(cls, average=True):
        global_t = cls.get()
        if global_t.total_time == 0 and global_t.start_time is not None:
            global_t.toc()
        print('Total time:', '\n'.join(global_t._get_lines(average=average)))

    @staticmethod
    def format(seconds):
        if seconds < 0.001:
            s, unit = '%.2f' % (seconds * 1000), 'ms'
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
