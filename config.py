import argparse
import os
import sys
import random

import pickle


class BaseConfigs:
    def parse_args(self, args, fail_if_missing=True):
        parser = argparse.ArgumentParser(description='%s settings' % type(self).__name__.split('Config')[0].capitalize())
        for k, v in vars(self).items():
            self._add_argument(parser, k, v, fail_if_missing=fail_if_missing)
        namespace = parser.parse_known_args(args)
        self.__dict__.update({k: v for k, v in vars(namespace[0]).items() if v is not None})
        self._postprocess_args(fail_if_missing)
        return namespace[1]

    def _add_argument(self, parser, param_name, param_value, fail_if_missing=True):
        if param_value is not None:
            parser_kwargs = {'dest': param_name}
            if type(param_value) == bool:
                parser_kwargs['action'] = 'store_%s' % str(not param_value).lower()
            else:
                parser_kwargs['type'] = type(param_value)
            parser.add_argument('--%s' % param_name, **parser_kwargs)

    def _postprocess_args(self, fail_if_missing):
        pass

    def __str__(self):
        s = []
        for k in sorted(self.__dict__.keys()):
            s += ['%-30s %s' % (k, self.__dict__[k])]
        return '\n'.join(s)


class ProgramConfig(BaseConfigs):
    def __init__(self):
        self.print_interval = 50
        self.log_interval = 100
        self.sync = False
        self.verbose = False

        self.debug = False
        self.monitor = False
        self.save_mem = False

        self.randomize = False

        self.resume = False

        self.model = None
        self.seenf = -1

        self.save_dir = ''

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
    def precomputed_data_dir_format(self):
        return os.path.join(cfg.program.cache_root, 'precomputed__%s_%s')

    @property
    def precomputed_feats_file_format(self):
        return os.path.join(self.cache_root, 'precomputed__%s_%s.h5')

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
    def load_train_final_output(self):
        return os.path.exists(self.saved_model_file)

    # @property
    # def resume(self):
    #     return os.path.exists(self.checkpoint_file)

    def _postprocess_args(self, fail_if_missing):
        self.save_dir = self.save_dir.rstrip('/')
        if '/' in self.save_dir:
            old_save_dir = self.save_dir
            self.save_dir = old_save_dir.split('/')[-1]
            self.model = self.model or old_save_dir.split('/')[-2]
            assert old_save_dir == self.output_path
        if fail_if_missing and self.model is None:
            raise ValueError('A model is required.')

    def _add_argument(self, parser, param_name, param_value, fail_if_missing=True):
        if param_name == 'model':
            from scripts.utils import get_all_models_by_name
            all_models_dict = get_all_models_by_name()
            all_models = set(all_models_dict.keys())
            parser.add_argument('--%s' % param_name, dest=param_name, type=str, choices=all_models)
        elif param_name == 'save_dir':
            parser.add_argument('--%s' % param_name, dest=param_name, type=str, required=fail_if_missing)
        else:
            super()._add_argument(parser, param_name, param_value)


class DataConfig(BaseConfigs):
    def __init__(self):
        self.pixel_mean = None
        self.pixel_std = None
        self.im_scale = None
        self.im_max_size = None

        self.filter_bg_only = False
        self.null_as_bg = False

        self.num_images = 0  # restrict the dataset to this number of images if > 0
        self.val_ratio = 0.1
        self.prinds = ''  # restrict the dataset to these predicates if not empty. FIXME fix interactions with seen file
        self.obinds = ''  # restrict the dataset to these objects if not empty

        self.union = True
        self.nw = 0

    @property
    def im_inds(self):
        return list(range(self.num_images)) if self.num_images > 0 else None

    @property
    def pred_inds(self):
        if not self.prinds:  # use all predicates
            return None
        try:  # case in which a single number is specified
            try:
                num_preds = int(self.prinds)
                pred_inds = list(range(num_preds))
            except ValueError:  # cannot cast to int
                num_possible_preds = 116  # FIXME magic constant
                num_preds = int(float(self.prinds) * num_possible_preds)
                pred_inds = [0] + sorted(random.sample(range(num_possible_preds), num_preds))
        except ValueError:  # cannot cast to number: a list has been specified
            pred_inds = sorted([int(pred_ind) for pred_ind in self.prinds.split(',')])
            if pred_inds[0] != 0:
                pred_inds = [0] + pred_inds
        return pred_inds

    @property
    def obj_inds(self):
        if not self.obinds:  # use all objects
            return None
        try:  # case in which a single number is specified
            num_objs = int(self.obinds)
            obj_inds = list(range(num_objs))
        except ValueError:  # cannot cast to int: a list has been specified
            obj_inds = sorted([int(obj_ind) for obj_ind in self.obinds.split(',')])
        return obj_inds


class ModelConfig(BaseConfigs):
    def __init__(self):
        self.rcnn_arch = 'e2e_mask_rcnn_R-50-C4_2x'
        self.mask_resolution = None
        self.hum_thr = 0.7
        self.obj_thr = 0.3

        self.dropout = 0.5
        self.repr_dim = 1024

        # BG
        self.filter = False

        # ZS
        self.attw = False
        self.oprior = False
        self.oscore = False
        self.softl = 0.0
        self.hoi_backbone = ''  # Path to the model final file, e.g. 'output/base/2019-06-05_17-43-04_vanilla/final.tar'

        self.aereg = 0.0
        self.regsmall = False

        # ZS GCN
        self.vv = False
        self.large = False

        # Predict action or interaction?
        self.phoi = False


class OptimizerConfig(BaseConfigs):
    def __init__(self):
        # Optimiser parameters
        self.adam = False
        self.momentum = 0.9
        self.l2_coeff = 1e-4
        self.grad_clip = 5.0
        self.num_epochs = 10

        # Learning rate parameters. Use gamma > 0 to enable decay at the specified interval
        self.lr = 1e-3
        self.lr_gamma = 0.0
        self.lr_decay_period = 4

        # Batch parameters
        self.group = False  # group HOIs belonging to the same image
        self.ohtrain = False  # one-hot train for (inter)actions, as opposed to multi-label
        self.img_batch_size = 8  # only used when grouping
        self.hoi_batch_size = 64
        self.hoi_bg_ratio = 3

        # Loss parameters
        self.margin = 0.0
        self.bg_coeff = 1.0
        self.fl_gamma = 0.0  # gamma in focal loss


class Configs:
    """
    @type program: ProgramConfig
    @type data: DataConfig
    @type model: ModelConfig
    @type opt: OptimizerConfig
    """
    program = ProgramConfig()
    data = DataConfig()
    model = ModelConfig()
    opt = OptimizerConfig()

    @classmethod
    def parse_args(cls, fail_if_missing=True, reset=False):
        args = sys.argv
        for k, v in sorted(cls.__dict__.items()):
            if isinstance(v, BaseConfigs):
                if reset:
                    v.__init__()
                args = v.parse_args(args, fail_if_missing=fail_if_missing)
        if args[1:]:
            # Detectron configurations should not be changed.
            raise ValueError('Invalid arguments: %s.' % ' '.join(args[1:]))
        sys.argv = sys.argv[:1]

        try:
            cls.init_detectron_cfgs()
        except ModuleNotFoundError:
            print('Detectron module not found')

    @classmethod
    def init_detectron_cfgs(cls):
        import lib.detection.wrappers as pydet
        if pydet.cfg_from_file is None:
            assert pydet.cfg is None and pydet.assert_and_infer_cfg is None
            cls.model.mask_resolution = 14  # FIXME magic constant
            raise ModuleNotFoundError()
        else:
            assert pydet.cfg is not None and pydet.assert_and_infer_cfg is not None

        cfg_file = 'pydetectron/configs/baselines/%s.yaml' % cls.model.rcnn_arch

        print("Loading Detectron's configs from {}.".format(cfg_file))
        pydet.cfg_from_file(cfg_file)
        pydet.cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
        pydet.cfg.MODEL.NUM_CLASSES = len(pydet.COCO_CLASSES)
        pydet.assert_and_infer_cfg()

        cls.data.pixel_mean = pydet.cfg.PIXEL_MEANS
        cls.data.im_scale = pydet.cfg.TEST.SCALE
        cls.data.im_max_size = pydet.cfg.TEST.MAX_SIZE
        cls.model.mask_resolution = pydet.cfg.MRCNN.RESOLUTION

    @classmethod
    def print(cls):
        s = []
        for k, v in sorted(cls.__dict__.items()):
            if isinstance(v, BaseConfigs):
                str_cfg = str(v)
                if str_cfg.strip():
                    s += ['{0}\n{1} configs\n{0}'.format('=' * 70, type(v).__name__.split('Config')[0].capitalize())]
                    s += [str_cfg, '']
        print('\n'.join(s))

    @classmethod
    def to_dict(cls):
        d = {}
        for k, v in vars(cls).items():
            if isinstance(v, BaseConfigs):
                d[k] = vars(v)
        return d

    @classmethod
    def save(cls, file_path=None):
        file_path = file_path or cls.program.config_file
        with open(file_path, 'wb') as f:
            pickle.dump(cls.to_dict(), f)

    @classmethod
    def load(cls, file_path=None, program=True, data=True, model=True, opt=True, reset=False):
        file_path = file_path or cls.program.config_file
        with open(file_path, 'rb') as f:
            d = pickle.load(f)
        if program:
            output_path = cls.program.output_path
            save_dir = cls.program.save_dir
            cls.program.__dict__.update(d['program'])
            cls.program.save_dir = save_dir
            assert cls.program.output_path.rstrip('/') == output_path.rstrip('/'), (cls.program.output_path, output_path)
        if data:
            cls.data.__dict__.update(d['data'])
        if model:
            cls.model.__dict__.update(d['model'])
        if opt:
            cls.opt.__dict__.update(d['opt'])


# Alias
cfg = Configs


def main():
    # print('Default configs')
    # Configs.print()

    sys.argv += ['--sync', '--model', 'nmotifs', '--save_dir', 'blabla', '--bn', '--grad_clip', '1.5']
    Configs.parse_args()
    # print('Updated with args:', sys.argv)
    Configs.print()


if __name__ == '__main__':
    main()
