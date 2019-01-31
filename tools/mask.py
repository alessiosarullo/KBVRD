import os

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


def save_plot(img, filename):
    plt.imshow(img)
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    # plt.show()
    plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=150)


def main(recolor=True, lw=10):
    data_dir = os.path.join('data', 'sample')
    img = np.array(Image.open(os.path.join(data_dir, 'im_friends.jpg')).load_raw('RGB'))
    print(img.shape)

    img_with_bbs = img.copy()
    union_bb = np.array([img.shape[0], img.shape[1], 0, 0])
    for i, name in enumerate(['joey', 'rachel', 'union']):

        if name == 'union':
            bb = union_bb
            mask_color = [0, 0, 0]
        else:
            # Read the mask
            mask_img = Image.open(os.path.join(data_dir, 'im_friends_mask_%s.jpg' % name))
            mask_img_color = np.array(mask_img.load_raw('RGB'))

            # Convert to grayscale and find bounding box enclosing the mask
            mask = 1 - np.round(np.array(mask_img.load_raw('L')) / 255)
            idx_y, idx_x = np.where(mask > 0)
            bb = [min(idx_y), min(idx_x), max(idx_y), max(idx_x)]

            # Find mask color, or assign a new one and save the recolored mask
            if recolor:
                mask_color = [0] * i + [255] + [0] * (2 - i)
                mask_img_color[idx_y, idx_x] = mask_color
                save_plot(mask_img_color, os.path.join(data_dir, 'im_friends_mask_recolor_%s.png' % name))
            else:
                mask_colors, color_counts = np.unique(mask_img_color[idx_y, idx_x, :], return_counts=True, axis=0)
                mask_color = mask_colors[np.argmax(color_counts)]

            # Update the union bounding box
            union_bb[:2] = np.minimum(union_bb[:2], bb[:2])
            union_bb[2:] = np.maximum(union_bb[2:], bb[2:])

            # Filter the image through the mask and save the result
            mask_img_patch = img[:mask.shape[0], :mask.shape[1], :] * mask[:, :, None].astype(np.bool)
            mask_img_patch[mask == 0] = 255
            save_plot(mask_img_patch, os.path.join(data_dir, 'im_friends_patch_mask_%s.png' % name))

            # Save the bounding box on the image
            img_with_bbs[bb[0]:bb[2], bb[1]:bb[1] + lw] = mask_color
            img_with_bbs[bb[0]:bb[2], bb[3] - lw:bb[3]] = mask_color
            img_with_bbs[bb[0]:bb[0] + lw, bb[1]:bb[3]] = mask_color
            img_with_bbs[bb[2] - lw:bb[2], bb[1]:bb[3]] = mask_color

        # Save bounding box only
        bb_only_img = np.ones_like(img) * 255
        bb_only_img[bb[0]:bb[2], bb[1]:bb[3]] = mask_color
        bb_only_img[bb[0] + lw:bb[2] - lw, bb[1] + lw:bb[3] - lw] = 255
        save_plot(bb_only_img, os.path.join(data_dir, 'im_friends_bbonly_%s.png' % name))

        # Save the image with the bounding box
        img_with_bb = img.copy()
        img_with_bb[bb[0]:bb[2], bb[1]:bb[1] + lw] = mask_color
        img_with_bb[bb[0]:bb[2], bb[3] - lw:bb[3]] = mask_color
        img_with_bb[bb[0]:bb[0] + lw, bb[1]:bb[3]] = mask_color
        img_with_bb[bb[2] - lw:bb[2], bb[1]:bb[3]] = mask_color
        save_plot(img_with_bb, os.path.join(data_dir, 'im_friends_bb_%s.png' % name))

        # Filter the image through the bounding box and save the result
        bb_img_patch = bb_only_img
        bb_img_patch[bb[0] + lw:bb[2] - lw, bb[1] + lw:bb[3] - lw] = \
            img[bb[0] + lw:bb[2] - lw, bb[1] + lw:bb[3] - lw]
        save_plot(bb_img_patch, os.path.join(data_dir, 'im_friends_patch_bb_%s.png' % name))

        # Save the cropped-out image patch
        img_crop = img[bb[0]:bb[2], bb[1]:bb[3]]
        save_plot(img_crop, os.path.join(data_dir, 'im_friends_crop_bb_%s.png' % name))

    save_plot(img_with_bbs, os.path.join(data_dir, 'im_friends_bbs.png'))


if __name__ == '__main__':
    main()
