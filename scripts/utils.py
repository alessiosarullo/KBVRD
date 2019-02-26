import numpy as np
import time
import torch
from typing import Dict


class Timer:
    global_timer = None
    gpu_sync = False

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
                timer = timer.sub_timers[subt]
            else:
                timer = timer.sub_timers.setdefault(subt, Timer())
        return timer

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

    def spent(self, average=True):
        return self.total_time / (self.num_instances if average else 1)

    def _get_lines(self, average):
        sep = ' ' * 4
        s = ['%s (x%d)' % (self.format(self.spent(average=average)), self.num_instances)]
        for k, v in self.sub_timers.items():
            sub_s = v._get_lines(average)
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
            s, unit = '%d:%02d' % divmod(seconds/60, 60), 'h'
        return '%s%s' % (s, unit)


def print_params(model, breakdown=False):
    """
    Prints parameters of a model
    """

    def _format(_n):
        if _n < 10 ** 3:
            return '%d' % _n
        elif _n < 10 ** 6:
            return '%.1fk' % (_n / 10 ** 3)
        else:
            return '%.1fM' % (_n / 10 ** 6)

    modules = {'RCNN': {}, 'Object branch': {}, 'Spatial branch': {}, 'Human-Object-Interaction branch': {}, 'Other': {}}
    for p_name, p in model.named_parameters():
        if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):

            p_name_root = p_name.split('.')[0]
            if 'rcnn' in p_name_root:
                module = 'RCNN'
            elif p_name_root.startswith('obj'):
                module = 'Object branch'
            elif 'spatial' in p_name_root:
                module = 'Spatial branch'
            elif p_name_root.startswith('hoi'):
                module = 'Human-Object-Interaction branch'
            else:
                module = 'Other'
            modules[module][p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
    if not modules['Other']:
        del modules['Other']

    total_params, trainable_params = 0, 0
    summary_strings, strings = [], []
    for module, mod_data in modules.items():
        module_tot = sum([s[1] for s in mod_data.values()])
        module_trainable = sum([s[1] for s in mod_data.values() if s[2]])
        total_params += module_tot
        trainable_params += module_trainable

        summary_strings.append(' - %6s (%6s) %s' % (_format(module_tot), _format(module_trainable), module))

        strings.append('### %s' % module)
        for p_name, (size, prod, p_req_grad) in sorted(mod_data.items(), key=lambda x: -x[1][1]):
            strings.append("{:<100s}: {:<16s}({:8d})".format(p_name, '[{}]'.format(','.join(size)), prod))

    s = '\n{0}\n{1} total parameters ({2} trainable ones):\n{3}\n{4}\n{0}'.format('#' * 100,
                                                                                  _format(total_params),
                                                                                  _format(trainable_params),
                                                                                  '\n'.join(summary_strings),
                                                                                  '\n'.join(strings) if breakdown else '')
    print(s, flush=True)
