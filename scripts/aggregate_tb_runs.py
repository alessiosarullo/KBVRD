import argparse
import os

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboardX import SummaryWriter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('fname')
    parser.add_argument('runs', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    fname = args.fname
    runs = args.runs

    print(fname)
    print(runs)
    os.makedirs(fname, exist_ok=True)

    summary_iterators = [EventAccumulator(os.path.join(p, 'tboard/Test')).Reload() for p in runs]
    tags = summary_iterators[0].Tags()['scalars']
    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags

    values_per_tag = {}
    steps = []
    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            values_per_tag.setdefault(tag, []).append([e.value for e in events])
            assert len({e.step for e in events}) == 1
            if len(values_per_tag.keys()) == 1:
                steps.append(events[0].step)
    steps = np.array(steps)

    aggr_ops = {'mean': np.mean,
                'std': np.std}
    for aggr_op_name, aggr_op in aggr_ops.items():
        tblogger = SummaryWriter(os.path.join(fname, 'tboard/Test', aggr_op_name))
        for tag, values in values_per_tag.items():
            aggr_value_per_timestep = aggr_op(np.array(values), axis=1)
            for i, aggr_value in enumerate(aggr_value_per_timestep):
                tblogger.add_scalar(tag, aggr_value, global_step=steps[i])


if __name__ == '__main__':
    main()
