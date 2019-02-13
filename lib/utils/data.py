from enum import Enum

import numpy as np
import torch
from PIL import ImageOps

from config import Configs as cfg
from lib.pydetectron_api.wrappers import prep_im_for_blob, cfg as pydet_cfg  # FIXME merge configs


class Splits(Enum):
    TRAIN = 'train'
    VAL = 'val'
    TEST = 'test'


class Minibatch:
    def __init__(self, training):
        self.training = training

        self.imgs = []
        self.img_infos = []
        self.img_fns = []

        if training:
            self.gt_boxes = []
            self.gt_box_classes = []
            self.gt_box_im_ids = []
            self.gt_inters = []
            self.gt_inters_im_ids = []

    def append(self, ex):
        self.imgs += [ex['img']]
        self.img_infos += [np.array([*ex['img_size'], ex['scale']], dtype=np.float32)]
        self.img_fns += [ex['fn']]

        if self.training:
            self.gt_boxes += [ex['gt_boxes'] * ex['scale']]
            self.gt_box_classes += [ex['gt_box_classes']]
            self.gt_inters += [ex['gt_inters']]

            self.gt_box_im_ids += [np.ones_like(ex['gt_box_classes']) * len(self.gt_box_im_ids)]
            self.gt_inters_im_ids += [np.ones_like(ex['gt_inters']) * len(self.gt_inters_im_ids)]

    def vectorize(self, device):
        self.imgs = _im_list_to_4d_tensor(self._to_tensor(self.imgs, device=device))  # 4D NCHW tensor
        self.img_infos = np.stack(self.img_infos, axis=0)
        assert self.imgs.shape[0] == self.img_infos.shape[0]

        if self.training:
            self.gt_box_im_ids = np.concatenate(self.gt_box_im_ids, axis=0)
            self.gt_inters_im_ids = np.concatenate(self.gt_inters_im_ids, axis=0)
            self.gt_boxes = np.concatenate(self.gt_boxes, axis=0)
            self.gt_box_classes = np.concatenate(self.gt_box_classes, axis=0)
            self.gt_inters = np.concatenate(self.gt_inters, axis=0)

            assert len(self.gt_boxes) == len(self.gt_box_classes) == len(self.gt_box_im_ids)
            assert len(self.gt_inters) == len(self.gt_inters_im_ids)

    @classmethod
    def collate(cls, examples, training):
        minibatch = cls(training)
        for ex in examples:
            minibatch.append(ex)
        minibatch.vectorize(device=torch.device('cuda'))  # FIXME magic constant
        return minibatch

    @staticmethod
    def _to_tensor(values, device, concat=False):
        tensor_list = [torch.tensor(v, device=device) for v in values]
        if concat:
            return torch.cat(tensor_list, dim=0)
        else:
            return tensor_list


class SquarePad(object):
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
    ims, im_scale = prep_im_for_blob(im, pydet_cfg.PIXEL_MEANS, [pydet_cfg.TEST.SCALE], pydet_cfg.TEST.MAX_SIZE)
    assert len(ims) == 1
    processed_im = np.transpose(ims[0], axes=(2, 0, 1))  # to CHW
    im_scale = np.squeeze(im_scale)
    return processed_im, im_scale


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