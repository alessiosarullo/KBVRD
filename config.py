import argparse
import os
import sys

import lib.detection.wrappers as pydet


class BaseConfigs:
    def parse_args(self):
        args = self._get_arg_parser().parse_known_args()
        self.__dict__.update({k: v for k, v in vars(args[0]).items() if v is not None})

    def _get_arg_parser(self):
        parser = argparse.ArgumentParser(description='%s settings' % type(self).__name__.split('Config')[0].capitalize())
        for k, v in vars(self).items():
            parser_kwargs = {'dest': k}
            if type(v) == bool:
                parser_kwargs['action'] = 'store_%s' % str(not v).lower()
            else:
                parser_kwargs['type'] = type(v)
            parser.add_argument('--%s' % k, **parser_kwargs)
        return parser

    def __str__(self):
        s = []
        for k in sorted(self.__dict__.keys()):
            s += ['%-30s %s' % (k, self.__dict__[k])]
        return '\n'.join(s)


class ProgramConfig(BaseConfigs):
    def __init__(self):
        self.print_interval = 50
        self.randomize = False
        self.sync = False

        self.eval_only = False
        self.predcls = False

        self.save_dir = 'output'
        self.load_precomputed_feats = False

    @property
    def detectron_pretrained_file_format(self):
        return os.path.join('data', 'pretrained_model', '%s.pkl')

    @property
    def precomputed_feats_file_format(self):
        return os.path.join('cache', 'precomputed__%s.h5')

    @property
    def checkpoint_file(self):
        return os.path.join(self.save_dir, 'ckpt.tar')

    @property
    def saved_model_file(self):
        return os.path.join(self.save_dir, 'final.tar')

    @property
    def result_file_format(self):
        return os.path.join(self.save_dir, 'result_test_%s.pkl')

    @property
    def config_file(self):
        return os.path.join(self.save_dir, 'config.pkl')


class DataConfig(BaseConfigs):
    def __init__(self):
        self.pixel_mean = None
        self.pixel_std = None
        self.im_scale = None
        self.im_max_size = None

        self.flip_prob = 0.0

        self.num_images = 0  # restrict the dataset to this number of images if > 0
        self.prinds = ''  # restrict the dataset to these predicates if not empty
        self.obinds = ''  # restrict the dataset to these objects if not empty

    @property
    def im_inds(self):
        return list(range(self.num_images)) if self.num_images > 0 else None

    @property
    def pred_inds(self):
        if not self.prinds:  # use all predicates
            return None
        try:  # case in which a single number is specified
            num_preds = int(self.prinds)
            pred_inds = list(range(num_preds))
        except ValueError:  # cannot cast to int: a list has been specified
            pred_inds = sorted([int(pred_ind) for pred_ind in self.prinds.split(',')])
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


class OptimizerConfig(BaseConfigs):
    def __init__(self):
        self.use_adam = False
        self.learning_rate = 1e-3
        self.l2_coeff = 1e-4
        self.grad_clip = 5.0

        self.num_epochs = 10
        self.batch_size = 4


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
    def parse_args(cls):
        for k, v in sorted(cls.__dict__.items()):
            if isinstance(v, BaseConfigs):
                v.parse_args()
        cls.init_detectron_cfgs()
        # Detectron configurations override command line arguments. This is ok, since the model's configs should not be changed. TODO Raise a warning.
        # TODO (related to the above) when trying to set a parameter that defaults to None a useless error is printed. Say that this parameter
        #  cannot be set through command line

    @classmethod
    def init_detectron_cfgs(cls):
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


# Alias
cfg = Configs


def main():
    # print('Default configs')
    # Configs.print()

    sys.argv += ['--load_precomputed_feats']
    Configs.parse_args()
    # print('Updated with args:', sys.argv)
    Configs.print()


if __name__ == '__main__':
    main()
