import sys
import argparse
import os
import lib.pydetectron_api.wrappers as pydet


class BaseConfigs:
    def parse_args(self):
        args = self._get_arg_parser().parse_args()
        self.__dict__.update({k: v for k, v in vars(args).items() if v is not None})

    def _get_arg_parser(self):
        raise NotImplementedError

    def __str__(self):
        s = []
        for k in sorted(self.__dict__.keys()):
            s += ['%-30s %s' % (k, self.__dict__[k])]
        return '\n'.join(s)


class ProgramConfig(BaseConfigs):
    def __init__(self):
        self.print_interval = 10
        self.randomize = False

        self.save_dir = 'exp'
        self.detectron_pretrained_file_format = os.path.join('data', 'pretrained_model', '%s.pkl')

    def _get_arg_parser(self):
        parser = argparse.ArgumentParser(description='Program settings')
        return parser


class DataConfig(BaseConfigs):
    def __init__(self):
        self.pixel_mean = None
        self.pixel_std = None
        self.im_scale = None
        self.im_max_size = None

    def _get_arg_parser(self):
        parser = argparse.ArgumentParser(description='Data settings')
        return parser


class ModelConfig(BaseConfigs):
    def __init__(self):
        self.rcnn_arch = 'e2e_mask_rcnn_R-50-C4_2x'
        self.mask_resolution = None

    def _get_arg_parser(self):
        parser = argparse.ArgumentParser(description='Model settings')
        parser.add_argument('--rcnn_arch', dest='rcnn_arch', type=str,
                            help='Name of the RCNN architecture to use. Pretrained weights and configurations will be loaded according to this.')
        return parser


class OptimizerConfig(BaseConfigs):
    def __init__(self):
        self.use_adam = False
        self.learning_rate = 1e-3
        self.l2_coeff = 1e-4
        self.grad_clip = 5.0

        self.num_epochs = 10

    def _get_arg_parser(self):
        parser = argparse.ArgumentParser(description='Optimizer settings')
        return parser


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
                    s += ['{0}\n{1} configs\n{0}'.format('=' * 50, k.strip('_').capitalize())]
                    s += [str_cfg, '']
        print('\n'.join(s))


def main():
    print('Default configs')
    Configs.print()

    # sys.argv += ['--feats', '/path/to/feats']
    Configs.parse_args()
    print('Updated with args:', sys.argv)
    Configs.print()


if __name__ == '__main__':
    main()