from enum import Enum

import numpy as np
from PIL import ImageOps

from config import Configs as cfg
from lib.detection.wrappers import prep_im_for_blob


class Splits(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class SquarePad:
    def __call__(self, img):
        w, h = img.size
        pixel_mean = cfg.data.pixel_mean
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
    ims, im_scale = prep_im_for_blob(im, cfg.data.pixel_mean, [cfg.data.im_scale], cfg.data.im_max_size)
    assert len(ims) == 1
    processed_im = np.transpose(ims[0], axes=(2, 0, 1))  # to CHW
    im_scale = np.squeeze(im_scale)
    return processed_im, im_scale
