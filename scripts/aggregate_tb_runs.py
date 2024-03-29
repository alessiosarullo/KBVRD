import argparse
import os

import numpy as np
from tensorboardX import SummaryWriter

from lib.utils import get_runs_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    parser.add_argument('runs', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    fname = args.fname
    runs = args.runs

    # try:  # PyCharm debugging
    #     print('Starting remote debugging (resume from debug server)')
    #     import pydevd_pycharm
    #     pydevd_pycharm.settrace('130.88.195.105', port=16004, stdoutToServer=True, stderrToServer=True)
    #     print('Remote debugging activated.')
    # except:
    #     print('Remote debugging failed.')
    #     raise

    print(fname)
    print(runs)
    os.makedirs(fname, exist_ok=True)

    runs_data = get_runs_data(runs)
    for split in ['Train', 'Val', 'Test']:
        values_per_tag = runs_data[split]['values']
        steps = runs_data[split]['steps']

        aggr_ops = {'mean': np.mean}
        for aggr_op_name, aggr_op in aggr_ops.items():
            tblogger = SummaryWriter(os.path.join(fname, 'tboard', split, aggr_op_name))
            for tag, values in values_per_tag.items():
                aggr_value_per_timestep = aggr_op(values, axis=0)
                for i, aggr_value in enumerate(aggr_value_per_timestep):
                    tblogger.add_scalar(tag, aggr_value, global_step=steps[i])


if __name__ == '__main__':
    main()
