import argparse
import os
import pickle
import sys


class Configs:
    def __init__(self):

        ##########################################
        # General options                        #
        ##########################################

        # Program
        self.sync = False
        self.debug = False
        self.monitor = False
        self.save_memory = False
        self.nworkers = 0

        # Print
        self.print_interval = 50
        self.log_interval = 100  # Tensorboard
        self.verbose = False

        # Experiment
        self.randomize = False
        self.resume = False
        self.save_dir = ''
        self.seenf = -1

        ##########################################
        # Data options                           #
        ##########################################

        # Image
        self.pixel_mean = None
        self.pixel_std = None
        self.im_scale = None
        self.im_max_size = None

        # Null/background
        self.filter_bg_only = False
        self.null_as_bg = False

        # Dataset
        self.val_ratio = 0.1
        self.hico = False  # use HICO [True] or HICO-DET [False]

        ##########################################
        # Model options                          #
        ##########################################
        self.model = None
        self.phoi = False  # Predict action [False] or interaction [True]?

        # Detector
        self.rcnn_arch = 'e2e_mask_rcnn_R-50-C4_2x'
        self.mask_resolution = None
        self.hum_thr = 0.7
        self.obj_thr = 0.3

        # Architecture
        self.dropout = 0.5
        self.repr_dim = 1024

        # Loss
        self.fl_gamma = 0.0  # gamma in focal loss
        self.meanc = False  # mean or sum over classes for BCE loss?

        # HICO specific
        self.hico_lhard = False
        self.hico_zso1 = False
        self.hico_zso2 = False
        self.hico_zsa = False

        # BG specific
        self.filter = False

        # ZS specific
        self.attw = False
        self.oprior = False
        self.oscore = False
        self.hoi_backbone = ''  # Path to the model final file, e.g. 'output/base/2019-06-05_17-43-04_vanilla/final.tar'
        self.softl = 0.0
        self.nullzs = False
        self.lis = False

        # ZS GCN specific
        self.greg = 0.0
        self.greg_margin = 0.3
        self.vv = False
        self.puregc = False
        self.iso_null = False
        self.aggp = False
        self.hoigcn = False

        ##########################################
        # Optimiser options                      #
        ##########################################

        # Optimiser
        self.adam = False
        self.adamb1 = 0.9
        self.adamb2 = 0.999
        self.momentum = 0.9
        self.l2_coeff = 5e-4
        self.grad_clip = 5.0
        self.num_epochs = 10

        # Learning rate. A value of 0 means that option is disabled.
        self.lr = 1e-3
        self.lr_gamma = 0.1
        self.lr_decay_period = 0
        self.lr_warmup = 0
        self.c_lr_gcn = 0.0

        # Batch
        self.group = False  # group HOIs belonging to the same image
        self.ohtrain = False  # one-hot train for (inter)actions, as opposed to multi-label
        self.img_batch_size = 8  # only used when grouping
        self.batch_size = 64
        self.hoi_bg_ratio = 3

    @property
    def output_root(self):
        return 'output'

    @property
    def data_root(self):
        return 'data'

    @property
    def cache_root(self):
        return 'cache'

    @property
    def embedding_dir(self):
        return os.path.join(self.data_root, 'embeddings')

    @property
    def detectron_pretrained_file_format(self):
        return os.path.join(self.data_root, 'pretrained_model', '%s.pkl')

    @property
    def precomputed_feats_format(self):
        return os.path.join(self.cache_root, 'precomputed_%s__%s_%s.h5')

    @property
    def output_path(self):
        return os.path.join(self.output_root, self.model, self.save_dir)

    @property
    def config_file(self):
        return os.path.join(self.output_path, 'config.pkl')

    @property
    def checkpoint_file(self):
        return os.path.join(self.output_path, 'ckpt.tar')

    @property
    def watched_values_file(self):
        return os.path.join(self.output_path, 'watched.tar')

    @property
    def saved_model_file(self):
        return os.path.join(self.output_path, 'final.tar')

    @property
    def prediction_file(self):
        return os.path.join(self.output_path, 'prediction_test.pkl')

    @property
    def eval_res_file(self):
        return os.path.join(self.output_path, 'eval_test.pkl')

    @property
    def ds_inds_file(self):
        return os.path.join(self.output_path, 'ds_inds.pkl')

    @property
    def active_classes_file(self):
        assert self.seenf >= 0
        return os.path.join('zero-shot_inds', f'seen_inds_{self.seenf}.pkl.push')

    @property
    def tensorboard_dir(self):
        return os.path.join(self.output_path, 'tboard')

    @property
    def res_stats_path(self):
        return os.path.join(self.output_path, 'res_stats')

    @property
    def eval_only(self):
        return os.path.exists(self.saved_model_file) and not self.resume

    def parse_args(self, fail_if_missing=True, reset=False):
        args = sys.argv
        if reset:
            self.__init__()

        parser = argparse.ArgumentParser(description='Settings')
        for k, v in vars(self).items():
            self._add_argument(parser, k, v, fail_if_missing=fail_if_missing)
        namespace = parser.parse_known_args(args)
        self.__dict__.update({k: v for k, v in vars(namespace[0]).items() if v is not None})
        self._postprocess_args(fail_if_missing)

        args = namespace[1]
        if args[1:]:
            # Invalid options: either unknown or ones initialised as None, which are Detectron's and should not be changed.
            raise ValueError('Invalid arguments: %s.' % ' '.join(args[1:]))
        sys.argv = sys.argv[:1]

        try:
            self.init_detectron_cfgs()
        except ModuleNotFoundError:
            print('Detectron module not found')

    def _add_argument(self, parser, param_name, param_value, fail_if_missing=True):
        if param_name == 'model':
            from scripts.utils import get_all_models_by_name
            all_models_dict = get_all_models_by_name()
            all_models = set(all_models_dict.keys())
            parser.add_argument('--%s' % param_name, dest=param_name, type=str, choices=all_models)
        elif param_name == 'save_dir':
            parser.add_argument('--%s' % param_name, dest=param_name, type=str, required=fail_if_missing)
        else:
            if param_value is not None:
                parser_kwargs = {'dest': param_name}
                if type(param_value) == bool:
                    parser_kwargs['action'] = 'store_%s' % str(not param_value).lower()
                else:
                    parser_kwargs['type'] = type(param_value)
                parser.add_argument('--%s' % param_name, **parser_kwargs)

    def _postprocess_args(self, fail_if_missing):
        self.save_dir = self.save_dir.rstrip('/')
        if '/' in self.save_dir:
            old_save_dir = self.save_dir
            self.save_dir = old_save_dir.split('/')[-1]
            self.model = self.model or old_save_dir.split('/')[-2]
            assert old_save_dir == self.output_path
        if fail_if_missing and self.model is None:
            raise ValueError('A model is required.')

    def init_detectron_cfgs(self):
        import lib.detection.wrappers as pydet
        if pydet.cfg_from_file is None:
            assert pydet.cfg is None and pydet.assert_and_infer_cfg is None
            self.mask_resolution = 14  # FIXME magic constant
            raise ModuleNotFoundError()
        else:
            assert pydet.cfg is not None and pydet.assert_and_infer_cfg is not None

        cfg_file = f'pydetectron/configs/baselines/{self.rcnn_arch}.yaml'
        print(f"Loading Detectron's configs from {cfg_file}.")
        pydet.cfg_from_file(cfg_file)
        pydet.cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
        pydet.cfg.MODEL.NUM_CLASSES = len(pydet.COCO_CLASSES)
        pydet.assert_and_infer_cfg()

        self.pixel_mean = pydet.cfg.PIXEL_MEANS
        self.im_scale = pydet.cfg.TEST.SCALE
        self.im_max_size = pydet.cfg.TEST.MAX_SIZE
        self.mask_resolution = pydet.cfg.MRCNN.RESOLUTION

    def save(self, file_path=None):
        file_path = file_path or self.config_file
        with open(file_path, 'wb') as f:
            pickle.dump(vars(self), f)

    def load(self, file_path=None):
        file_path = file_path or self.config_file
        with open(file_path, 'rb') as f:
            d = pickle.load(f)

        # Save options that should not be loaded
        output_path = self.output_path
        save_dir = self.save_dir
        resume = self.resume
        num_epochs = self.num_epochs

        # Load
        self.__dict__.update(d)

        # Restore
        self.save_dir = save_dir
        self.resume = resume
        assert self.output_path.rstrip('/') == output_path.rstrip('/'), (self.output_path, output_path)
        if resume:
            self.num_epochs += num_epochs

    def print(self):
        print(str(self), '\n')

    def __str__(self):
        s = []
        for k in sorted(self.__dict__.keys()):
            s += ['%-30s %s' % (k, self.__dict__[k])]
        return '{0}\nConfigs\n{0}\n{1}\n{0}\n{0}'.format('=' * 70, '\n'.join(s))


# Instantiate
cfg = Configs()  # type: Configs


def test():
    # print('Default configs')
    # Configs.print()

    sys.argv += ['--sync', '--model', 'hicobase', '--save_dir', 'blabla']
    cfg.parse_args()
    # print('Updated with args:', sys.argv)
    cfg.print()


if __name__ == '__main__':
    test()
