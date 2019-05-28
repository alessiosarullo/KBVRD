from enum import Enum

import cv2
import numpy as np
import torch
from PIL import ImageOps

from config import cfg
from lib.stats.utils import Timer


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


class GTEntry:
    def __init__(self, idx_in_split, img_id, filename, split):
        self.index = idx_in_split
        self.id = img_id
        self.filename = filename
        self.split = split

        self.img_size = None
        self.scale = None

        self.image = None
        self.gt_boxes = None
        self.gt_obj_classes = None
        self.gt_hois = None


class PrecomputedExample:
    def __init__(self, idx_in_split, img_id, filename, split):
        self.index = idx_in_split
        self.id = img_id
        self.filename = filename
        self.split = split

        self.img_size = None
        self.scale = None

        self.precomp_boxes_ext = None
        self.precomp_box_feats = None
        self.precomp_masks = None

        self.precomp_hoi_infos = None
        self.precomp_hoi_union_boxes = None
        self.precomp_hoi_union_feats = None

        self.precomp_box_labels = None
        self.precomp_action_labels = None


class Minibatch:
    # TODO refactor in list[Example], then merged when called vectorize()
    def __init__(self):

        self.img_infos = []
        self.other_ex_data = []

        self.imgs = []
        self.gt_boxes = []
        self.gt_obj_classes = []
        self.gt_box_im_ids = []
        self.gt_hois = []
        self.gt_hoi_im_ids = []

    def append(self, ex: GTEntry):
        self.img_infos += [np.array([*ex.img_size, ex.scale], dtype=np.float32)]

        self.other_ex_data += [{'index': ex.index,
                                'id': ex.id,
                                'fn': ex.filename,
                                'split': ex.split,
                                'im_size': ex.img_size,  # this won't be changed
                                'im_scale': ex.scale,
                                }]

        self.imgs += [ex.image]
        self.gt_boxes += [ex.gt_boxes * ex.scale]
        self.gt_obj_classes += [ex.gt_obj_classes]
        self.gt_hois += [ex.gt_hois]

        self.gt_box_im_ids += [np.full_like(ex.gt_obj_classes, fill_value=len(self.gt_box_im_ids))]
        self.gt_hoi_im_ids += [np.full(ex.gt_hois.shape[0], fill_value=len(self.gt_hoi_im_ids), dtype=np.int)]

    def vectorize(self, device):
        assert all([len(v) > 0 for k, v in self.__dict__.items() if k.startswith('gt_')])

        self.imgs = _im_list_to_4d_tensor([torch.tensor(v, device=device) for v in self.imgs])  # 4D NCHW tensor

        img_infos = np.stack(self.img_infos, axis=0)
        img_infos[:, 0] = self.imgs.shape[2]
        img_infos[:, 1] = self.imgs.shape[3]
        self.img_infos = torch.tensor(img_infos, dtype=torch.float32, device=device)

        assert self.imgs.shape[0] == self.img_infos.shape[0]

        self.gt_box_im_ids = np.concatenate(self.gt_box_im_ids, axis=0)
        self.gt_boxes = np.concatenate(self.gt_boxes, axis=0)
        self.gt_obj_classes = np.concatenate(self.gt_obj_classes, axis=0)
        self.gt_hoi_im_ids = np.concatenate(self.gt_hoi_im_ids, axis=0)
        self.gt_hois = np.concatenate(self.gt_hois, axis=0)
        assert len(self.gt_boxes) == len(self.gt_obj_classes) == len(self.gt_box_im_ids)
        assert len(self.gt_hois) == len(self.gt_hoi_im_ids)

    @classmethod
    def collate(cls, examples):
        minibatch = cls()
        for ex in examples:
            minibatch.append(ex)
        minibatch.vectorize(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return minibatch


class PrecomputedMinibatch:
    def __init__(self):
        self.img_infos = []
        self.other_ex_data = []
        self.pc_boxes_ext = []
        self.pc_box_feats = []
        self.pc_masks = []
        self.pc_ho_infos = []
        self.pc_ho_union_boxes = []
        self.pc_ho_union_feats = []
        self.pc_box_labels = []
        self.pc_action_labels = []

    def append(self, ex: PrecomputedExample):
        im_id_in_batch = len(self.img_infos)
        self.img_infos += [np.array([*ex.img_size, ex.scale], dtype=np.float32)]

        self.other_ex_data += [{'index': ex.index,
                                'id': ex.id,
                                'fn': ex.filename,
                                'split': ex.split,
                                'im_size': ex.img_size,  # this won't be changed
                                'im_scale': ex.scale,
                                }]

        boxes_ext = ex.precomp_boxes_ext
        if boxes_ext is not None:
            boxes_ext[:, 0] = im_id_in_batch
            self.pc_boxes_ext += [boxes_ext]
            self.pc_box_feats += [ex.precomp_box_feats]
            self.pc_masks += [ex.precomp_masks]

            self.pc_box_labels += [ex.precomp_box_labels]

            hoi_infos = ex.precomp_hoi_infos
            if hoi_infos is not None:
                num_boxes = sum([boxes.shape[0] for boxes in self.pc_boxes_ext[:-1]])
                hoi_infos[:, 0] = im_id_in_batch
                hoi_infos[:, 1:] += num_boxes
                self.pc_ho_infos += [hoi_infos]
                self.pc_ho_union_boxes += [ex.precomp_hoi_union_boxes]
                self.pc_ho_union_feats += [ex.precomp_hoi_union_feats]

                self.pc_action_labels += [ex.precomp_action_labels]

    def vectorize(self, device):
        for k, v in self.__dict__.items():
            if k.startswith('pc_') and ('label' not in k):
                if not v:
                    v = [np.empty(0)]
                self.__dict__[k] = np.concatenate(v, axis=0)

        assert self.pc_boxes_ext.shape[0] == self.pc_box_feats.shape[0] == self.pc_masks.shape[0]
        assert self.pc_ho_infos.shape[0] == self.pc_ho_union_boxes.shape[0] == self.pc_ho_union_feats.shape[0]

        img_infos = np.stack(self.img_infos, axis=0)
        img_infos[:, 0] = max(img_infos[:, 0])
        img_infos[:, 1] = max(img_infos[:, 1])
        self.img_infos = torch.tensor(img_infos, dtype=torch.float32, device=device)

        if self.pc_box_labels[0] is None:
            assert all([l is None for l in self.pc_box_labels])
            assert all([l is None for l in self.pc_action_labels])
            self.pc_box_labels = self.pc_action_labels = None
        else:
            assert all([l is not None for l in self.pc_box_labels])
            assert all([l is not None for l in self.pc_action_labels])
            assert len(self.pc_box_labels) == len(self.pc_action_labels) == self.img_infos.shape[0], \
                (len(self.pc_box_labels), len(self.pc_action_labels), self.img_infos.shape[0])
            self.pc_box_labels = np.concatenate(self.pc_box_labels, axis=0)
            self.pc_action_labels = np.concatenate(self.pc_action_labels, axis=0)
            assert self.pc_boxes_ext.shape[0] == self.pc_box_labels.shape[0]
            assert self.pc_ho_infos.shape[0] == self.pc_action_labels.shape[0]

    @classmethod
    def collate(cls, examples):
        minibatch = cls()
        for ex in examples:
            minibatch.append(ex)
        minibatch.vectorize(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        return minibatch


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
        ex = dataset.get_img_entry(i, read_img=False)  # type: GTEntry
        gt_hois = ex.gt_hois
        objs = ex.gt_obj_classes[gt_hois[:, 2]]
        assert np.all(ex.gt_obj_classes[gt_hois[:, 0]] == dataset.human_class)
        for o, pred in zip(objs, gt_hois[:, 1]):
            counts[o, pred] += 1
    return counts
