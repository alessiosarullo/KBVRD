from enum import Enum

import cv2
import numpy as np
import torch
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


def _im_list_to_4d_tensor(ims, use_fpn=False):
    """
    :param ims [list(Tensor)]: List of N color images to concatenate.
    :param use_fpn:
    :return: im_tensor [Tensor, Nx3xHxW]: Tensor in NCHW format. Width and height are the maximum across all images and 0 padding is added at the
                end when necessary.
    """
    # TODO doc is incomplete

    assert isinstance(ims, list)
    max_shape = _get_max_shape([im.shape[1:] for im in ims], stride=32. if use_fpn else 1)  # FIXME magic constant stride

    num_images = len(ims)
    im_tensor = ims[0].new_zeros((num_images, 3, max_shape[0], max_shape[1]), dtype=torch.float32)
    for i in range(num_images):
        im = ims[i]
        im_tensor[i, :, :im.shape[1], :im.shape[2]] = im
    return im_tensor


def _get_max_shape(im_shapes, stride=1.):
    """
    Calculate max spatial size (h, w) for batching given a list of image shapes. Takes into account FPN coarsest stride if using it.
    """
    max_shape = np.array(im_shapes).max(axis=0)
    assert max_shape.size == 2
    # Pad the image so they can be divisible by `stride`
    max_shape[0] = int(np.ceil(max_shape[0] / stride) * stride)
    max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)
    return max_shape


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
          - Convert to float32
          - Subtract per-channel pixel mean
          - Rescale to each of the specified target size (capped at max_size)
        Returns a list of transformed images, one for each target size. Also returns
        the scale factors that were used to compute each returned image.
        """

    im = im.astype(np.float32, copy=False)
    im -= cfg.data.pixel_mean
    im_scale = calculate_im_scale(im.shape[:2])
    im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_im = np.transpose(im_resized, axes=(2, 0, 1))  # to CHW
    return processed_im, im_scale


def calculate_im_scale(im_size):
    """ Calculate target resize scale. """
    max_size = cfg.data.im_max_size
    target_size = cfg.data.im_scale

    im_size_min = np.min(im_size)
    im_size_max = np.max(im_size)
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:  # Prevent the biggest axis from being more than max_size
        im_scale = float(max_size) / float(im_size_max)
    return im_scale


def get_counts(dataset):
    counts = np.zeros((dataset.num_object_classes, dataset.num_predicates), dtype=np.int64)
    for i in range(len(dataset)):
        ex = dataset.get_img_entry(i, read_img=False)  # type: Example
        gt_hois = ex.gt_hois
        objs = ex.gt_obj_classes[gt_hois[:, 2]]
        assert np.all(ex.gt_obj_classes[gt_hois[:, 0]] == dataset.human_class)
        for o, pred in zip(objs, gt_hois[:, 1]):
            counts[o, pred] += 1
    return counts
