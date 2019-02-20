"""
Hides this ugliness and avoids IDE's "unresolved import" errors
"""

import os.path as osp
import sys


sys.path.insert(0, osp.abspath(osp.join('pydetectron', 'lib')))
# Note: fixing these imports doesn't work if Detectron's ones stay the same, because they end up looking at different configs (for some reason).
from core.config import \
    cfg as _pydet_cfg, \
    cfg_from_file as _pydet_cfg_from_file, \
    assert_and_infer_cfg as _pydet_assert_and_infer_cfg

from core.test import \
    _get_blobs as _pydet_get_blobs, \
    _get_rois_blob as _pydet_get_rois_blob, \
    _add_multilevel_rois_for_test as _pydet_add_multilevel_rois_for_test, \
    box_utils as _pydet_box_utils, \
    segm_results as _pydet_segm_results

import datasets.dummy_datasets as _pydet_dummy_datasets

from model.nms.nms_gpu import nms_gpu as _pydet_nms_gpu

from modeling.model_builder import Generalized_RCNN as _PydetGeneralized_RCNN

import utils.misc as _pydet_misc_utils
import utils.vis as _pydet_vis_utils
from utils.timer import Timer as _PydetTimer
from utils.detectron_weight_helper import load_detectron_weight as _pydet_load_detectron_weight
from utils.blob import \
    prep_im_for_blob as _pydet_prep_im_for_blob, \
    get_image_blob as _pydet_get_image_blob

# Configs
cfg = _pydet_cfg
cfg_from_file = _pydet_cfg_from_file
assert_and_infer_cfg = _pydet_assert_and_infer_cfg
dummy_datasets = _pydet_dummy_datasets
COCO_CLASSES = dummy_datasets.get_coco_dataset().classes

# Needed for detection
_get_blobs = _pydet_get_blobs
_get_rois_blob = _pydet_get_rois_blob
_add_multilevel_rois_for_test = _pydet_add_multilevel_rois_for_test
nms_gpu = _pydet_nms_gpu
box_utils = _pydet_box_utils
Timer = _PydetTimer

# Imported for other scripts
segm_results = _pydet_segm_results
Generalized_RCNN = _PydetGeneralized_RCNN
misc_utils = _pydet_misc_utils
vis_utils = _pydet_vis_utils
load_detectron_weight = _pydet_load_detectron_weight
prep_im_for_blob = _pydet_prep_im_for_blob
get_image_blob = _pydet_get_image_blob

sys.path.remove(osp.abspath(osp.join('pydetectron', 'lib')))
