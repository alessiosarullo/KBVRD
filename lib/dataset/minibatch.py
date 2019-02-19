import numpy as np
import torch

from scripts.utils import Timer


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

        # These will be empty if not using precomputed features
        self.pc_box_feats = []
        self.pc_boxes = []
        self.pc_box_scores = []
        self.box_im_ids = []
        self.box_pred_classes = []

    def append(self, ex):
        self.imgs += [ex['img']]
        self.img_infos += [np.array([*ex['img_size'], ex['scale']], dtype=np.float32)]
        self.img_fns += [ex['fn']]

        if self.training:
            self.gt_boxes += [ex['gt_boxes'] * ex['scale']]
            self.gt_box_classes += [ex['gt_box_classes']]
            self.gt_inters += [ex['gt_inters']]

            self.gt_box_im_ids += [np.full_like(ex['gt_box_classes'], fill_value=len(self.gt_box_im_ids))]
            self.gt_inters_im_ids += [np.full(ex['gt_inters'].shape[0], fill_value=len(self.gt_inters_im_ids), dtype=np.int)]

        try:
            self.pc_box_feats += [ex['box_feats']]
            self.pc_boxes += [ex['boxes']]
            self.pc_box_scores += [ex['box_scores']]
            self.box_im_ids += [np.full(ex['boxes'].shape[0], fill_value=len(self.box_im_ids))]
            self.box_pred_classes += [ex['box_pred_classes']]
        except KeyError:
            pass

    def vectorize(self, device):
        self.imgs = _im_list_to_4d_tensor(self._to_tensor(self.imgs, device=device))  # 4D NCHW tensor

        img_infos = np.stack(self.img_infos, axis=0)
        im_scales = img_infos[:, 2]
        im_infos = np.concatenate([np.tile(self.imgs.shape[2:], reps=[im_scales.size, 1]), im_scales[:, None]], axis=1)
        self.img_infos = torch.tensor(im_infos, dtype=torch.float32, device=device)

        assert self.imgs.shape[0] == self.img_infos.shape[0]

        if self.training:
            self.gt_box_im_ids = np.concatenate(self.gt_box_im_ids, axis=0)
            self.gt_inters_im_ids = np.concatenate(self.gt_inters_im_ids, axis=0)
            self.gt_boxes = np.concatenate(self.gt_boxes, axis=0)
            self.gt_box_classes = np.concatenate(self.gt_box_classes, axis=0)
            self.gt_inters = np.concatenate(self.gt_inters, axis=0)

            assert len(self.gt_boxes) == len(self.gt_box_classes) == len(self.gt_box_im_ids)
            assert len(self.gt_inters) == len(self.gt_inters_im_ids)

        assert len(self.pc_box_feats) == len(self.pc_boxes) == len(self.pc_box_scores) == len(self.box_im_ids) == len(self.box_pred_classes)
        if self.pc_box_feats:
            self.box_im_ids = np.concatenate(self.box_im_ids, axis=0)
            self.pc_boxes = np.concatenate(self.pc_boxes, axis=0)
            self.pc_box_scores = np.concatenate(self.pc_box_scores, axis=0)
            self.pc_box_feats = self._to_tensor(self.pc_box_feats, device, concat=True)
            self.box_pred_classes = np.concatenate(self.box_pred_classes, axis=0)
        else:
            self.pc_box_feats = self.pc_boxes = self.pc_box_scores = self.box_im_ids = self.box_pred_classes = None

    @classmethod
    def collate(cls, examples, training):
        Timer.get('Epoch', 'Collate').tic()
        minibatch = cls(training)
        for ex in examples:
            minibatch.append(ex)
        minibatch.vectorize(device=torch.device('cuda'))  # FIXME magic constant
        Timer.get('Epoch', 'Collate').toc()
        return minibatch

    @staticmethod
    def _to_tensor(values, device, concat=False):
        tensor_list = [torch.tensor(v, device=device) for v in values]
        if concat:
            return torch.cat(tensor_list, dim=0)
        else:
            return tensor_list


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