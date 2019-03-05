from enum import Enum

import cv2
import numpy as np

from PIL import ImageOps

from config import cfg


class Splits(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class SquarePad:
    def __call__(self, img, pixel_mean):
        w, h = img.size
        img_padded = ImageOps.expand(img, border=(0, 0, max(h - w, 0), max(w - h, 0)),
                                     fill=(int(pixel_mean[0] * 256), int(pixel_mean[1] * 256), int(pixel_mean[2] * 256)))
        return img_padded


def preprocess_img(im):
    """
    Preprocess an image to be used as an input by normalising, converting to float and rescaling to all scales specified in the configurations (
    rescaling is capped). NOTE: so far only one scale can be specified.
    :param im [image]: A BGR image in HWC format. Images read with OpenCV's `imread` satisfy these conditions.
    :return: - processed_im [image]: The transformed image, in CHW format with BGR channels.
             - im_scale [scalar]: The scale factor that was used.
    """

    # TODO fix docs
    """Prepare an image for use as a network input blob. Specially:
          - Subtract per-channel pixel mean
          - Convert to float32
          - Rescale to each of the specified target size (capped at max_size)
        Returns a list of transformed images, one for each target size. Also returns
        the scale factors that were used to compute each returned image.
        """
    max_size = cfg.data.im_max_size
    target_size = cfg.data.im_scale

    im = im.astype(np.float32, copy=False)
    im -= cfg.data.pixel_mean
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    # Calculate target resize scale
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:  # Prevent the biggest axis from being more than max_size
        im_scale = float(max_size) / float(im_size_max)

    im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_im = np.transpose(im_resized, axes=(2, 0, 1))  # to CHW
    return processed_im, im_scale