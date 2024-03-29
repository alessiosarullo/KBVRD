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
        self.eval_interval = 0
        self.verbose = False

        # Experiment
        self.randomize = False
        self.resume = False
        self.save_dir = ''
        self.seenf = -1

        ##########################################
        # Data options                           #
        ##########################################

        # Null/background
        self.filter_bg_only = False
        self.null_as_bg = False

        # Dataset
        self.val_ratio = 0.1
        self.hicodet = False  # if True, use HICO-DET. In this case, the next option is ignored.
        self.vghoi = False  # use HICO [False] or VGHOI [True]

        ##########################################
        # Model options                          #
        ##########################################
        self.model = None
        self.phoi = False  # Predict action [False] or interaction [True]?

        # Detector. The output dim is usually hardcoded in their files (e.g., `ResNet_roi_conv5_head_for_masks()`), so I can't read it from configs.
        self.rcnn_arch = 'e2e_mask_rcnn_R-50-C4_2x'
        self.rcnn_output_dim = {'e2e_mask_rcnn_R-50-C4_2x': 2048}[self.rcnn_arch]
        self.hum_thr = 0.7
        self.obj_thr = 0.3

        # Architecture
        self.dropout = 0.5
        self.repr_dim = 1024

        # Loss
        self.fl_gamma = 0.0  # gamma in focal loss
        self.meanc = False  # mean or sum over classes for BCE loss?
        self.csp = False  # Use cost-sensitive coefficients for positive examples

        # HICO specific
        self.train_null = False
        # Soft labels, score, loss, regularisation, cost-sensitive loss
        self.osl = 0.0
        self.osc = 1.0
        self.olc = 1.0
        self.opr = 0.0
        self.ocs = False
        #
        self.asl = 0.0
        self.asc = 1.0
        self.alc = 1.0
        self.apr = 0.0
        self.acs = False
        #
        self.hsl = 0.0
        self.hlc = 0.0
        self.hsc = 0.0
        self.hpr = 0.0
        self.hcs = False
        self.rl_no_norm = False
        self.gc = False
        self.hoigcn = False
        # Kato specific
        self.katopadj = False
        self.katopgc = False
        self.katoconstz = False

        # ZS specific
        self.hoi_backbone = ''  # Path to the model final file, e.g. 'output/base/2019-06-05_17-43-04_vanilla/final.tar'
        self.lis = False
        self.slpure = False

        # ZS GCN specific
        self.gconly = False
        self.link_null = False
        self.puregc = False
        self.greg = 0.0
        self.greg_margin = 0.3
        self.gcldim = 1024
        self.gcrdim = 1024

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
        if self.vghoi:
            return os.path.join('zero-shot_inds', 'seen_inds_vghoi.pkl.push')
        else:
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
            save_dir_parts = self.save_dir.split('/')
            if save_dir_parts[0] == self.output_root:
                assert len(save_dir_parts) >= 3
                old_save_dir = self.save_dir
                self.save_dir = os.path.join(*save_dir_parts[2:])
                self.model = self.model or save_dir_parts[1]
                assert old_save_dir == self.output_path
        if fail_if_missing and self.model is None:
            raise ValueError('A model is required.')

    def init_detectron_cfgs(self):
        import lib.detection.wrappers as pydet
        if pydet.cfg_from_file is None:
            assert pydet.cfg is None and pydet.assert_and_infer_cfg is None
            self.mask_resolution = 14
            raise ModuleNotFoundError()
        else:
            assert pydet.cfg is not None and pydet.assert_and_infer_cfg is not None

        cfg_file = f'pydetectron/configs/baselines/{self.rcnn_arch}.yaml'
        print(f"Loading Detectron's configs from {cfg_file}.")
        pydet.cfg_from_file(cfg_file)
        pydet.cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
        pydet.cfg.MODEL.NUM_CLASSES = len(pydet.COCO_CLASSES)
        pydet.assert_and_infer_cfg()

        # self.pixel_mean = pydet.cfg.PIXEL_MEANS
        # self.im_scale = pydet.cfg.TEST.SCALE
        # self.im_max_size = pydet.cfg.TEST.MAX_SIZE

    def save(self, file_path=None):
        file_path = file_path or self.config_file
        with open(file_path, 'wb') as f:
            pickle.dump(vars(self), f)

    def load(self, file_path=None):
        cfg_file_path = file_path or self.config_file
        with open(cfg_file_path, 'rb') as f:
            d = pickle.load(f)

        if file_path is None:
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
        else:
            self.__dict__.update(d)
            self.resume = False

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

    # sys.argv += ['--sync', '--model', 'hicoall', '--save_dir', 'blabla/2019-24-12_param1/RUN1']
    # sys.argv += ['--save_dir', 'output/hicoall/blabla/2019-24-12_param1/RUN1']
    # sys.argv += ['--save_dir', 'hicoall/blabla/2019-24-12_param1/RUN1']  # this should fail
    sys.argv += ['--save_dir', 'output/hicoall/']  # this too
    cfg.parse_args()
    # print('Updated with args:', sys.argv)
    cfg.print()
    print(cfg.output_path)


if __name__ == '__main__':
    test()
