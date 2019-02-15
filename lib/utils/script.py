import numpy as np


def print_para(model, breakdown=False):
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
    if modules['Other'] == 0:
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
