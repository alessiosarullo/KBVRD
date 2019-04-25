import matplotlib
import numpy as np
import torch
from matplotlib import gridspec
from matplotlib import pyplot as plt

try:
    matplotlib.use('Qt5Agg')
except ImportError:
    pass


def main():
    files = ['analysis/output/plots/run_hoi_2019-04-22_17-04-13_b64_tboard_Test-tag-M-mAP.csv',
             'analysis/output/plots/run_embsim_2019-04-22_19-53-50_vanilla_tboard_Test-tag-M-mAP.csv']

    font = {'family': 'normal',
            'size': 18}
    matplotlib.rc('font', **font)

    ax = plt.gca()
    for f in files:
        d = np.loadtxt(f, skiprows=1, delimiter=',')
        plt.plot(np.arange(d.shape[0]) + 1, d[:, 2])
        ax.set_xticks(np.arange(d.shape[0]) + 1)
    ax.set_ylabel('Mean average precision')
    ax.set_xlabel('Epoch')
    plt.legend(['Baseline', 'w/ knowledge'])
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
