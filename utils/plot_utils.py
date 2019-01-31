import numpy as np
from matplotlib import pyplot as plt


def plot_mat(conf_mat, xticklabels, yticklabels):
    lfsize = 8
    plt.figure(figsize=(16, 9))
    ax = plt.gca()
    ax.matshow(conf_mat, cmap=plt.get_cmap('jet'), vmin=0, vmax=1)

    y_tick_labels = [l.replace('_', ' ') for l in yticklabels]
    y_ticks = np.arange(len(y_tick_labels))

    maj_ticks = y_ticks[::2]
    maj_tick_labels = y_tick_labels[::2]
    ax.set_yticks(maj_ticks)
    ax.set_yticklabels(maj_tick_labels)
    ax.tick_params(axis='y', which='major', left=True, labelleft=True, right=True, labelright=False, labelsize=lfsize)

    min_ticks = y_ticks[1::2]
    min_tick_labels = y_tick_labels[1::2]
    ax.set_yticks(min_ticks, minor=True)
    ax.set_yticklabels(min_tick_labels, minor=True)
    ax.tick_params(axis='y', which='minor', left=True, labelleft=False, right=True, labelright=True, labelsize=lfsize)

    x_tick_labels = [l.replace('_', ' ') for l in xticklabels]
    x_ticks = np.arange(len(x_tick_labels))

    maj_ticks = x_ticks[::2]
    maj_tick_labels = x_tick_labels[::2]
    ax.set_xticks(maj_ticks)
    ax.set_xticklabels(maj_tick_labels, rotation=45, ha='left', rotation_mode='anchor')
    ax.tick_params(axis='x', which='major', top=True, labeltop=True, bottom=True, labelbottom=False, labelsize=lfsize)

    min_ticks = x_ticks[1::2]
    min_tick_labels = x_tick_labels[1::2]
    ax.set_xticks(min_ticks, minor=True)
    ax.set_xticklabels(min_tick_labels, minor=True, rotation=45, ha='right', rotation_mode='anchor')
    ax.tick_params(axis='x', which='minor', top=True, labeltop=False, bottom=True, labelbottom=True, labelsize=lfsize)

    plt.tight_layout()
    plt.show()
