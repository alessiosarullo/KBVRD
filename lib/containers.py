import numpy as np
import torch

from lib.stats.utils import Timer


class Example:
    def __init__(self, idx_in_split, img_id, img_fn,
                 gt_boxes, gt_obj_classes, gt_hois,
                 image, img_size, img_scale_factor, flipped,):
        self.index = idx_in_split
        self.id = img_id
        self.fn = img_fn

        self.image = image
        self.img_size = img_size
        self.scale = img_scale_factor
        self.flipped = flipped

        self.gt_boxes = gt_boxes
        self.gt_obj_classes = gt_obj_classes
        self.gt_hois = gt_hois

        self.precomputed_boxes = None
        self.precomputed_box_feats = None
        self.precomputed_obj_scores = None
        self.precomputed_obj_classes = None

    @property
    def has_precomputed(self):
        if self.precomputed_boxes is not None:
            assert self.precomputed_obj_scores is not None and \
                   self.precomputed_box_feats is not None and \
                   self.precomputed_obj_classes is not None
            return True
        else:
            assert self.precomputed_obj_scores is None and self.precomputed_box_feats is None and self.precomputed_obj_classes is None
            return False


class Minibatch:
    # TODO refactor in list[Example], then merged when called vectorize()
    def __init__(self):

        self.imgs = []
        self.img_infos = []
        self.other_ex_data = []

        self.gt_boxes = []
        self.gt_obj_classes = []
        self.gt_box_im_ids = []
        self.gt_hois = []
        self.gt_hoi_im_ids = []

        # These will be empty if not using precomputed features
        self.pc_box_feats = []
        self.pc_boxes = []
        self.pc_box_scores = []
        self.box_im_ids = []
        self.box_pred_classes = []

    def append(self, ex: Example):
        self.imgs += [ex.image]
        self.img_infos += [np.array([*ex.img_size, ex.scale], dtype=np.float32)]

        self.other_ex_data += [{'index': ex.index,
                                'id': ex.id,
                                'fn': ex.fn,
                                'flipped': ex.flipped
                                }]

        self.gt_boxes += [ex.gt_boxes * ex.scale]
        self.gt_obj_classes += [ex.gt_obj_classes]
        self.gt_hois += [ex.gt_hois]

        self.gt_box_im_ids += [np.full_like(ex.gt_obj_classes, fill_value=len(self.gt_box_im_ids))]
        self.gt_hoi_im_ids += [np.full(ex.gt_hois.shape[0], fill_value=len(self.gt_hoi_im_ids), dtype=np.int)]

        if ex.has_precomputed:
            self.pc_box_feats += [ex.precomputed_box_feats]
            self.pc_boxes += [ex.precomputed_boxes]
            self.pc_box_scores += [ex.precomputed_obj_scores]
            self.box_im_ids += [np.full(ex.precomputed_boxes.shape[0], fill_value=len(self.box_im_ids))]
            self.box_pred_classes += [ex.precomputed_obj_classes]

    def vectorize(self, device):
        self.imgs = _im_list_to_4d_tensor(self._to_tensor(self.imgs, device=device))  # 4D NCHW tensor

        img_infos = np.stack(self.img_infos, axis=0)
        im_scales = img_infos[:, 2]
        im_infos = np.concatenate([np.tile(self.imgs.shape[2:], reps=[im_scales.size, 1]), im_scales[:, None]], axis=1)
        self.img_infos = torch.tensor(im_infos, dtype=torch.float32, device=device)
        assert self.imgs.shape[0] == self.img_infos.shape[0]

        self.gt_box_im_ids = np.concatenate(self.gt_box_im_ids, axis=0)
        self.gt_boxes = np.concatenate(self.gt_boxes, axis=0)
        self.gt_obj_classes = np.concatenate(self.gt_obj_classes, axis=0)
        self.gt_hoi_im_ids = np.concatenate(self.gt_hoi_im_ids, axis=0)
        self.gt_hois = np.concatenate(self.gt_hois, axis=0)
        assert len(self.gt_boxes) == len(self.gt_obj_classes) == len(self.gt_box_im_ids)
        assert len(self.gt_hois) == len(self.gt_hoi_im_ids)

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
    def collate(cls, examples):
        Timer.get('Collate').tic()
        minibatch = cls()
        for ex in examples:
            minibatch.append(ex)
        minibatch.vectorize(device=torch.device('cuda'))  # FIXME magic constant
        Timer.get('Collate').toc()
        return minibatch

    @staticmethod
    def _to_tensor(values, device, concat=False):
        tensor_list = [torch.tensor(v, device=device) for v in values]
        if concat:
            return torch.cat(tensor_list, dim=0)
        else:
            return tensor_list


class Prediction:
    def __init__(self, obj_im_inds, obj_boxes, obj_scores, hoi_img_inds, ho_pairs, hoi_scores):
        self.obj_im_inds = obj_im_inds  # type: np.ndarray
        self.obj_boxes = obj_boxes  # type: np.ndarray
        self.obj_scores = obj_scores  # type: np.ndarray
        self.hoi_img_inds = hoi_img_inds  # type: np.ndarray
        self.ho_pairs = ho_pairs  # type: np.ndarray
        self.hoi_score_distributions = hoi_scores  # type: np.ndarray

    @property
    def hoi_classes(self):
        return self.hoi_score_distributions.argmax(axis=1)

    @classmethod
    def from_dict(cls, prediction_dict):
        p = Prediction(obj_im_inds=None, obj_boxes=None, obj_scores=None, hoi_img_inds=None, ho_pairs=None, hoi_scores=None)
        assert set(vars(p).keys()) == set(prediction_dict.keys())
        p.__dict__.update(prediction_dict)
        return p

    def is_complete(self):
        complete = True
        for v in vars(self).values():
            complete = complete and v is not None
        return complete


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
