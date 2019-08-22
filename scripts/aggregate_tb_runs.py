import os
import argparse

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('runs', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    runs = args.runs

    print(runs)

    summary_iterators = [EventAccumulator(os.path.join(p, 'tboard/Test')).Reload() for p in runs]
    tags = summary_iterators[0].Tags()['scalars']
    for it in summary_iterators:
        assert it.Tags()['scalars'] == tags
    out = {}
    for tag in tags:
        for events in zip(*[acc.Scalars(tag) for acc in summary_iterators]):
            assert len(set(e.step for e in events)) == 1
            out.setdefault(tag, []).append([e.value for e in events])


if __name__ == '__main__':
    main()
