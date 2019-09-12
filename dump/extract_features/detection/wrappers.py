"""
Hides this ugliness and avoids IDE's "unresolved import" errors
"""

import os.path as osp
import sys

sys.path.insert(0, osp.abspath(osp.join('pydetectron', 'lib')))

# These are the "safe" ones (they don't depend on having CUDA or compiled code)
try:
    import datasets.dummy_datasets as dummy_datasets
    dummy_datasets = dummy_datasets  # This fixes an "unresolved import" error
    COCO_CLASSES = dummy_datasets.get_coco_dataset().classes
except ImportError:
    raise

try:
    # Note: fixing these imports doesn't work if Detectron's ones stay the same, because they end up looking at different configs (for some reason).
    from core.config import cfg, cfg_from_file, assert_and_infer_cfg

    # from core.test import _get_blobs, _get_rois_blob, _add_multilevel_rois_for_test, box_utils, segm_results
    from core.test import im_detect_bbox, box_utils

    # from model.nms.nms_gpu import nms_gpu

    from modeling.model_builder import Generalized_RCNN

    # import utils.misc as misc_utils
    # from utils.timer import Timer
    from utils.detectron_weight_helper import load_detectron_weight
    # from utils.blob import prep_im_for_blob, get_image_blob

except ImportError:
    cfg = cfg_from_file = assert_and_infer_cfg = None
    box_utils = im_detect_bbox = None
    Generalized_RCNN = None
    load_detectron_weight = None
finally:
    sys.path.remove(osp.abspath(osp.join('pydetectron', 'lib')))
