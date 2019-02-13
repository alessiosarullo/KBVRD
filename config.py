import sys
import argparse


class BaseConfigs:
    def parse_args(self):
        args = self.get_arg_parser().parse_args()
        self.__dict__.update({k: v for k, v in vars(args).items() if v is not None})

    def get_arg_parser(self):
        raise NotImplementedError

    def __str__(self):
        s = []
        for k in sorted(self.__dict__.keys()):
            s += ['%-30s %s' % (k, self.__dict__[k])]
        return '\n'.join(s)


class ProgramConfig(BaseConfigs):
    def __init__(self):
        self.print_interval = 10
        self.save_dir = 'exp'
        self.randomize = False

    def get_arg_parser(self):
        pass


class DataConfig(BaseConfigs):
    def __init__(self):
        self.pretrained_features = None
        self.pixel_mean = None
        self.pixel_std = None
        self.im_scale = None

        # # Normalisation values used in NeuralMotifs
        # self.pixel_mean = [0.485, 0.456, 0.406]
        # self.pixel_std = [0.229, 0.224, 0.225]
        # self.im_scale = 600

    def get_arg_parser(self):
        parser = argparse.ArgumentParser(description='Data settings')
        # parser.add_argument('--feats', dest='pretrained_features', type=str,
        #                     help='If specified, absolute path to the file contained Mask-RCNN pretrained features.')
        return parser


class ModelConfig(BaseConfigs):
    def __init__(self):
        self.rcnn_arch = None

    def get_arg_parser(self):
        parser = argparse.ArgumentParser(description='Model settings')
        parser.add_argument('--rcnn_arch', dest='rcnn_arch', type=str, required=True,
                            help='Name of the RCNN architecture to use. Pretrained weights and configurations will be loaded according to this.')
        return parser


class OptimizerConfig(BaseConfigs):
    def __init__(self):
        self.use_adam = False
        self.learning_rate = 1e-3
        self.l2_coeff = 1e-4
        self.grad_clip = 5.0

        self.num_epochs = 10

    def get_arg_parser(self):
        pass


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
        cls.data.get_arg_parser()
        cls.model.get_arg_parser()
        cls.opt.get_arg_parser()

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

    sys.argv += ['--feats', '/path/to/feats']
    Configs.parse_args()
    print('Updated with args:', sys.argv)
    Configs.print()


if __name__ == '__main__':
    main()
