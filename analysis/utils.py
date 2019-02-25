import numpy as np
from matplotlib import pyplot as plt
import os


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


# TODO
def vis_one_image(
        im, im_name, output_dir, boxes, segms=None, keypoints=None, thresh=0.9,
        kp_thresh=2, dpi=200, box_alpha=0.0, dataset=None, show_class=False,
        ext='png'):
    raise NotImplementedError
    """Visual debugging of detections."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return

    dataset_keypoints, _ = keypoint_utils.get_keypoints()
    kp_lines = kp_connections(dataset_keypoints)
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kp_lines) + 2)]

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

        print(dataset.classes[classes[i]], score)
        # show box (off by default, box_alpha=0.0)
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1],
                          fill=False, edgecolor='g',
                          linewidth=0.5, alpha=box_alpha))

        if show_class:
            ax.text(
                bbox[0], bbox[1] - 2,
                get_class_string(classes[i], score, dataset),
                fontsize=3,
                family='serif',
                bbox=dict(
                    facecolor='g', alpha=0.4, pad=0, edgecolor='none'),
                color='white')

        # show mask
        if segms is not None and len(segms) > i:
            img = np.ones(im.shape)
            color_mask = color_list[mask_color_id % len(color_list), 0:3]
            mask_color_id += 1

            w_ratio = .4
            for c in range(3):
                color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
            for c in range(3):
                img[:, :, c] = color_mask[c]
            e = masks[:, :, i]

            _, contour, hier = cv2.findContours(
                e.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

            for c in contour:
                polygon = Polygon(
                    c.reshape((-1, 2)),
                    fill=True, facecolor=color_mask,
                    edgecolor='w', linewidth=1.2,
                    alpha=0.5)
                ax.add_patch(polygon)

        # show keypoints
        if keypoints is not None and len(keypoints) > i:
            kps = keypoints[i]
            plt.autoscale(False)
            for l in range(len(kp_lines)):
                i1 = kp_lines[l][0]
                i2 = kp_lines[l][1]
                if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
                    x = [kps[0, i1], kps[0, i2]]
                    y = [kps[1, i1], kps[1, i2]]
                    line = ax.plot(x, y)
                    plt.setp(line, color=colors[l], linewidth=1.0, alpha=0.7)
                if kps[2, i1] > kp_thresh:
                    ax.plot(
                        kps[0, i1], kps[1, i1], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)
                if kps[2, i2] > kp_thresh:
                    ax.plot(
                        kps[0, i2], kps[1, i2], '.', color=colors[l],
                        markersize=3.0, alpha=0.7)

            # add mid shoulder / mid hip for better visualization
            mid_shoulder = (
                kps[:2, dataset_keypoints.index('right_shoulder')] +
                kps[:2, dataset_keypoints.index('left_shoulder')]) / 2.0
            sc_mid_shoulder = np.minimum(
                kps[2, dataset_keypoints.index('right_shoulder')],
                kps[2, dataset_keypoints.index('left_shoulder')])
            mid_hip = (
                kps[:2, dataset_keypoints.index('right_hip')] +
                kps[:2, dataset_keypoints.index('left_hip')]) / 2.0
            sc_mid_hip = np.minimum(
                kps[2, dataset_keypoints.index('right_hip')],
                kps[2, dataset_keypoints.index('left_hip')])
            if (sc_mid_shoulder > kp_thresh and
                    kps[2, dataset_keypoints.index('nose')] > kp_thresh):
                x = [mid_shoulder[0], kps[0, dataset_keypoints.index('nose')]]
                y = [mid_shoulder[1], kps[1, dataset_keypoints.index('nose')]]
                line = ax.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines)], linewidth=1.0, alpha=0.7)
            if sc_mid_shoulder > kp_thresh and sc_mid_hip > kp_thresh:
                x = [mid_shoulder[0], mid_hip[0]]
                y = [mid_shoulder[1], mid_hip[1]]
                line = ax.plot(x, y)
                plt.setp(
                    line, color=colors[len(kp_lines) + 1], linewidth=1.0,
                    alpha=0.7)

        output_name = os.path.basename(im_name) + '.' + ext
        fig.savefig(os.path.join(output_dir, '{}'.format(output_name)), dpi=dpi)
        plt.close('all')