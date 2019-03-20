import os

import cv2
import numpy as np
# # Use a non-interactive backend
# import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from lib.bbox_utils import rescale_masks_to_img


def plot_mat(conf_mat, xticklabels, yticklabels, axes=None):
    lfsize = 8
    if axes is None:
        plt.figure(figsize=(16, 9))
        ax = plt.gca()
    else:
        ax = axes
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
    if axes is None:
        plt.show()


def vis_one_image(im, boxes, box_classes, class_names, masks=None, union_boxes=None, thresh=0.9,
                  output_file_path=None, show_class=False, dpi=200, box_alpha=0.0, ext='png'):
    """Visual debugging of detections."""

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return

    color_list = colormap(rgb=True) / 255
    fig = plt.figure(frameon=False)
    fig.set_size_inches(im.shape[1] / dpi, im.shape[0] / dpi)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.axis('off')
    fig.add_axes(ax)
    ax.imshow(im)

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    mask_color_id = 0
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score < thresh:
            continue

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='g',
                          linewidth=0.5, alpha=box_alpha))

        if show_class:
            ax.text(
                bbox[0], bbox[1] - 2,
                class_names[box_classes[i]] + ' {:0.2f}'.format(score).lstrip('0'),
                fontsize=3,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if masks is not None:
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = masks[i, :, :]

            _, contour, hier = cv2.findContours(e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=color_mask,
                    edgecolor='w', linewidth=1.2,
                    alpha=0.5)
                ax.add_patch(polygon)

    for ub in union_boxes:
        ax.add_patch(
            plt.Rectangle((ub[0], ub[1]),
                          ub[2] - ub[0],
                          ub[3] - ub[1],
                          fill=False, edgecolor='r',
                          linewidth=0.3, alpha=box_alpha))

    if output_file_path is None:
        plt.show()
    else:
        fig.savefig(output_file_path + '.' + ext, dpi=dpi)
        plt.close('all')


def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list


def postprocess_for_visualisation(boxes_ext, masks, union_boxes, img_infos):
    assert img_infos.shape[0] == 1
    img_infos = img_infos[0].cpu().numpy()
    im_h, im_w = img_infos[:2].astype(np.int)
    im_scale = img_infos[2]
    boxes_ext = boxes_ext.cpu().numpy()
    masks = masks.cpu().numpy()

    box_classes = np.argmax(boxes_ext[:, 5:], axis=1)
    boxes = boxes_ext[:, 1:5] / im_scale
    boxes_with_scores = np.concatenate((boxes, boxes_ext[np.arange(boxes_ext.shape[0]), 5 + box_classes][:, None]), axis=1)
    masks = rescale_masks_to_img(masks, boxes, im_h, im_w)

    union_boxes = union_boxes / im_scale
    return boxes_with_scores, box_classes, masks, union_boxes
