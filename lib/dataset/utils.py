from enum import Enum

import cv2
import numpy as np
import torch
from PIL import ImageOps

from config import cfg
from lib.detection.wrappers import COCO_CLASSES


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

    im = im.astype(np.float32, copy=False)
    im -= cfg.pixel_mean
    im_scale = calculate_im_scale(im.shape[:2])
    im_resized = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    processed_im = np.transpose(im_resized, axes=(2, 0, 1))  # to CHW
    return processed_im, im_scale


def calculate_im_scale(im_size):
    """ Calculate target resize scale. """
    max_size = cfg.im_max_size
    target_size = cfg.im_scale

    im_size_min = np.min(im_size)
    im_size_max = np.max(im_size)
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:  # Prevent the biggest axis from being more than max_size
        im_scale = float(max_size) / float(im_size_max)
    return im_scale


def get_hico_to_coco_mapping(hico_objects, split_objects=None):
    if split_objects is None:
        split_objects = hico_objects
    coco_obj_to_idx = {('hair dryer' if c == 'hair drier' else c).replace(' ', '_'): i for i, c in COCO_CLASSES.items()}
    assert set(coco_obj_to_idx.keys()) - {'__background__'} == set(hico_objects)
    mapping = np.array([coco_obj_to_idx[obj] for obj in split_objects], dtype=np.int)
    return mapping


def interactions_to_mat(hois, hico, np2np=False):
    # Default is Torch to Torch
    if not np2np:
        hois_np = hois.detach().cpu().numpy()
    else:
        hois_np = hois
    all_hois = np.stack(np.where(hois_np > 0), axis=1)
    all_interactions = np.concatenate([all_hois[:, :1], hico.interactions[all_hois[:, 1], :]], axis=1)
    inter_mat = np.zeros((hois.shape[0], hico.num_objects, hico.num_actions))
    inter_mat[all_interactions[:, 0], all_interactions[:, 2], all_interactions[:, 1]] = 1
    if np2np:
        return inter_mat
    else:
        return torch.from_numpy(inter_mat).to(hois)


def get_hoi_adjacency_matrix(dataset, isolate_null=None):
    if isolate_null is None:
        isolate_null = not cfg.link_null
    interactions = dataset.full_dataset.interactions
    inter_obj_adj = np.zeros((dataset.full_dataset.num_interactions, dataset.full_dataset.num_objects))
    inter_obj_adj[np.arange(interactions.shape[0]), interactions[:, 0]] = 1

    inter_act_adj = np.zeros((dataset.full_dataset.num_interactions, dataset.full_dataset.num_actions))
    inter_act_adj[np.arange(interactions.shape[0]), interactions[:, 1]] = 1

    adj = inter_obj_adj @ inter_obj_adj.T + inter_act_adj @ inter_act_adj.T
    adj = torch.from_numpy(adj).clamp(max=1).float()

    if isolate_null:
        null_hois = np.flatnonzero(np.any(inter_act_adj[:, 1:], axis=1))
        adj[null_hois, :] = 0
        adj[:, null_hois] = 0
        return adj
    else:
        return adj


def get_noun_verb_adj_mat(dataset, isolate_null=None):
    if isolate_null is None:
        isolate_null = not cfg.link_null
    noun_verb_links = torch.from_numpy((dataset.full_dataset.op_pair_to_interaction >= 0).astype(np.float32))
    if isolate_null:
        noun_verb_links[:, 0] = 0
    return noun_verb_links
