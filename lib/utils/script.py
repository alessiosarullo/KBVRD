import numpy as np


def print_para(model):
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

    modules = {'RCNN': {}, 'Object stream': {}, 'Spatial stream': {}, 'Relationship stream': {}, 'Other': {}}
    for p_name, p in model.named_parameters():
        if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):

            p_name_root = p_name.split('.')[0]
            if 'rcnn' in p_name_root:
                module = 'RCNN'
            elif p_name_root.startswith('obj'):
                module = 'Object stream'
            elif 'spatial' in p_name_root:
                module = 'Spatial stream'
            elif p_name_root.startswith('rel'):
                module = 'Relationship stream'
            else:
                module = 'Other'
            modules[module][p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)

    total_params = {}
    strings = []
    for module, st in modules.items():
        total_params[module] = sum([s[1] for s in st.values()])
        strings.append('### %s' % module)
        for p_name, (size, prod, p_req_grad) in sorted(st.items(), key=lambda x: -x[1][1]):
            strings.append("{:<100s}: {:<16s}({:8d}) ({})".format(
                p_name, '[{}]'.format(','.join(size)), prod, 'grad' if p_req_grad else '    '
            ))

    print(total_params)
    s = '\n{}\n{} total parameters:\n{}\n{}'.format('#' * 50,
                                                    _format(sum(total_params.values())),
                                                    '\n'.join([' - %-15s: %s' % (module, _format(tp)) for module, tp in total_params.items()]),
                                                    '\n'.join(strings))
    print(s, flush=True)
