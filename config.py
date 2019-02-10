import sys
import argparse


class BaseConfigs:
    def parse_args(self):
        raise NotImplementedError

    def __str__(self):
        s = []
        for k in sorted(self.__dict__.keys()):
            s += ['%-30s %s' % (k, self.__dict__[k])]
        return '\n'.join(s)


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

    def parse_args(self):
        parser = argparse.ArgumentParser(description='Data settings')

        # Program arguments
        parser.add_argument('--feats', dest='pretrained_features', type=str,
                            help='If specified, absolute path to the file contained Mask-RCNN pretrained features.')

        args = parser.parse_args()
        self.__dict__.update({k: v for k, v in vars(args).items() if v is not None})


class ModelConfig(BaseConfigs):
    def __init__(self):
        pass

    def parse_args(self):
        pass


class OptimConfig(BaseConfigs):
    def __init__(self):
        pass

    def parse_args(self):
        pass


class Configs:
    """
    @type data: DataConfig
    @type model: ModelConfig
    @type optim: OptimConfig
    """
    data = DataConfig()
    model = ModelConfig()
    optim = OptimConfig()

    @classmethod
    def parse_args(cls):
        cls.data.parse_args()
        cls.model.parse_args()
        cls.optim.parse_args()

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
