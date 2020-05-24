import os
import pickle
import sys

import matplotlib.transforms as transforms
import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from sklearn.manifold import TSNE

from config import cfg
from lib.dataset.hico import HicoSplit, Hico
from lib.dataset.utils import Splits
from lib.models.abstract_model import AbstractModel
from scripts.utils import get_all_models_by_name


def save():
    cfg.parse_args(fail_if_missing=False)
    cfg.load()

    if cfg.seenf >= 0:
        inds_dict = pickle.load(open(cfg.seen_classes_file, 'rb'))
        act_inds = sorted(inds_dict[Splits.TRAIN.value]['act'].tolist())
        obj_inds = sorted(inds_dict[Splits.TRAIN.value]['obj'].tolist())
    else:
        obj_inds = act_inds = None

    splits = HicoSplit.get_splits(obj_inds=obj_inds, act_inds=act_inds)
    train_split, val_split, test_split = splits[Splits.TRAIN], splits[Splits.VAL], splits[Splits.TEST]

    # Model
    model = get_all_models_by_name()[cfg.model](train_split)  # type: AbstractModel

    ckpt = torch.load(cfg.saved_model_file, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'])

    model.eval()
    _, act_class_embs = model.gcn()
    act_class_embs = act_class_embs.cpu().numpy()

    os.makedirs(cfg.output_analysis_path, exist_ok=True)
    np.save(os.path.join(cfg.output_analysis_path, 'act_embs'), act_class_embs)


def show():
    exp = 'glove'
    fig_fn = f'/home/alex/Dropbox/PhD Docs/My stuff/Conferences/Second paper/' +\
             f'v8.4 (Submitted ECCV - Rebuttal long)/images/tsne{"_" if exp else ""}{exp}.png'

    groups = {}
    offsets = {}
    perplexity = 20.0
    np.random.seed(3)

    if exp == '':
        sys.argv[1:] = ['--save_dir', 'output/skzs/hico_zsk_gc_nobg/wemboo/2019-11-08_13-16-46_SINGLE']
        offsets = {
            1: [0, 0],  # adjust
            3: [-0.8, 0.3],  # block
            5: [0, -0.6],  # board
            6: [-1, -0.8],  # break
            7: [-1.6, -0.8],  # brush_with
            8: [-0.7, 0.2],  # buy
            9: [-1.05, 0.15],  # carry
            11: [-1, 0.1],  # chase
            12: [-0.9, 0.2],  # check
            15: [0.1, -0.3],  # cook
            16: [0, 0],  # cut
            17: [-1.2, -0.7],  # cut_with
            18: [-1.1, -0.8],  # direct
            19: [-1, 0.1],  # drag
            20: [0, -0.6],  # dribble
            21: [0, 0],  # drink_with
            22: [-0.9, 0.25],  # drive
            25: [-1, -0.8],  # eat_at
            27: [0, 0],  # feed
            28: [0, 0],  # fill
            29: [-0.8, 0],  # flip
            32: [0.1, -0.4],  # greet
            33: [0.1, -0.3],  # grind
            35: [-1.2, -0.6],  # herd
            38: [-1.8, -0.3],  # hop_on
            41: [0, 0],  # hunt
            43: [0.1, -0.6],  # install
            46: [-0.9, 0.1],  # kiss
            47: [-1.0, 0.1],  # lasso
            48: [-1.3, 0.1],  # launch
            49: [0, 0],  # lick
            50: [0, 0.1],  # lie_on
            52: [-1.2, -0.5],  # light
            53: [-1, -0.7],  # load
            54: [0, -0.6],  # lose
            55: [0.1, -0.3],  # make
            56: [0, 0],  # milk
            59: [0, -0.2],  # operate
            60: [0, -0.7],  # pack
            62: [-1, -0.8],  # park
            63: [0, -0.4],  # pay
            64: [-1.2, -0.5],  # peel
            66: [-1.1, -0.3],  # pick
            67: [0.1, -0.2],  # pick_up
            68: [0, -0.6],  # point
            69: [0, -0.6],  # pour
            73: [-1, 0],  # read
            74: [0, 0],  # release
            75: [0, -0.5],  # repair
            78: [0, 0],  # run
            79: [0, -0.6],  # sail
            80: [-1.8, -0.3],  # scratch
            81: [-0.4, 0.2],  # serve
            85: [0, 0],  # sip
            87: [0.1, -0.3],  # sit_on
            89: [0, 0],  # smell
            90: [0, -0.6],  # spin
            91: [0, -0.3],  # squeeze
            93: [0, -0.6],  # stand_on
            94: [-0.4, 0.3],  # stand_under
            95: [0, -0.3],  # stick
            97: [0.1, -0.1],  # stop_at
            98: [0, 0],  # straddle
            102: [0, -0.4],  # teach
            103: [-1.2, -0.8],  # text_on
            104: [-1.2, 0.2],  # throw
            106: [0, 0],  # toast
            107: [-1.2, -0.3],  # train
            111: [0, 0],  # wash
            112: [0, 0],  # watch
        }
        groups = {'A1': (['wield', 'break'],
                         {'w': 2.5, 'h': 2.3, 'a': 0, 'c': 'tab:orange', 'ox': 0.1, 'oy': -0.1, 'tox': -1.2, 'toy': -2.5}),
                  'A2': (['swing', 'point'],
                         {'w': 2.5, 'h': 2, 'a': 45, 'c': 'tab:orange', 'ox': 0.5, 'oy': 0, 'tox': -2., 'toy': -1.}),
                  'B1': (['adjust', 'tie', 'pull', 'wear'],
                         {'w': 4.5, 'h': 2., 'a': -75, 'c': 'tab:purple', 'ox': 0.4, 'oy': 0.3, 'tox': -1.5, 'toy': -1.5}),
                  'B2': (['assemble'],
                         {'w': 2.5, 'h': 1.5, 'a': 0, 'c': 'tab:purple', 'ox': 1, 'oy': 0.1, 'tox': 1., 'toy': 1.}),
                  }
        np.random.seed(4)
    elif exp == 'ext':
        sys.argv[1:] = ['--save_dir', 'output/skzs/hico_zsk_gc_nobg_ext2/wemboo/2019-10-23_13-53-06_RUN1']
    elif exp == 'ra':
        sys.argv[1:] = ['--save_dir', 'output/skzs/hico_zsk_gc_nobg_Ra/wemboo_Ra-10-03/2019-11-08_13-16-53_SINGLE']
        offsets = {
            1: [0, 0],  # adjust
            2: [0, -0.7],  # assemble
            3: [-1.4, -0.7],  # block
            5: [0, 0],  # board
            6: [-1.5, -0.4],  # break
            7: [-1.6, 0.3],  # brush_with
            8: [0, 0],  # buy
            9: [0, 0],  # carry
            11: [-1, 0.1],  # chase
            12: [0.1, -0.4],  # check
            16: [0, -0.4],  # cut
            17: [-1.2, -0.7],  # cut_with
            18: [0, 0],  # direct
            20: [0, -0.6],  # dribble
            21: [-1.5, -0.8],  # drink_with
            22: [-0.9, 0.25],  # drive
            24: [0, -0.4],  # eat
            25: [-1, -0.8],  # eat_at
            27: [-0.9, 0.1],  # feed
            28: [0, 0],  # fill
            29: [-0.8, 0],  # flip
            32: [-1.3, -0.7],  # greet
            33: [-0.9, -0.8],  # grind
            34: [0, -0.4],  # groom
            35: [-1.2, -0.3],  # herd
            38: [0, -0.3],  # hop_on
            41: [0, 0],  # hunt
            43: [0, -0.5],  # install
            46: [-1.1, -0.3],  # kiss
            47: [0.1, -0.3],  # lasso
            48: [-1.3, 0.1],  # launch
            49: [0, 0],  # lick
            50: [-1.5, -0.6],  # lie_on
            52: [-1.2, 0.1],  # light
            53: [-0.6, 0.1],  # load
            54: [0, -0.6],  # lose
            55: [-1, 0.2],  # make
            56: [0, 0],  # milk
            59: [0, -0.4],  # operate
            60: [0, 0],  # pack
            63: [0, 0],  # pay
            64: [-0.9, 0.1],  # peel
            66: [0, 0],  # pick
            67: [0, -0.5],  # pick_up
            68: [0, -0.6],  # point
            69: [-0.4, 0.1],  # pour
            70: [0, -0.6],  # pull
            71: [0, -0.7],  # push
            72: [-0.5, 0.1],  # race
            73: [0.1, -0.4],  # read
            74: [-1.2, -0.7],  # release
            78: [-0.7, 0],  # run
            79: [0, -0.6],  # sail
            80: [0, -0.6],  # scratch
            81: [-0.4, -0.8],  # serve
            85: [0.1, -0.4],  # sip
            87: [-1.0, 0.1],  # sit_on
            89: [-1.4, -0.5],  # smell
            90: [-1, 0.1],  # spin
            91: [0, -0.3],  # squeeze
            93: [0, 0],  # stand_on
            94: [0, 0],  # stand_under
            95: [0, -0.3],  # stick
            97: [0, 0],  # stop_at
            98: [0, 0],  # straddle
            102: [0, 0],  # teach
            103: [-1.9, -0.3],  # text_on
            106: [0, 0],  # toast
            107: [0, -0.3],  # train
            111: [0, 0],  # wash
            112: [0.1, -0.4],  # watch
            114: [0., -0.5],  # wear
        }
        groups = {'A': (['wield', 'break', 'swing', 'point'],
                        {'w': 5, 'h': 2.3, 'a': 45, 'c': 'tab:orange', 'ox': 0.2, 'oy': 0., 'tox': -1.7, 'toy': 0.5}),
                  # 'B1': (['adjust', 'tie', 'wear'],
                  #        {'w': 3.3, 'h': 2.3, 'a': 0, 'c': 'tab:purple', 'ox': 0.4, 'oy': 0., 'tox': -1.5, 'toy': -2.2}),
                  # 'B2': (['assemble', 'pull'],
                  #        {'w': 4., 'h': 2, 'a': 0, 'c': 'tab:purple', 'ox': 1, 'oy': -0.2, 'tox': 1., 'toy': 1.}),
                  'B': (['adjust', 'tie', 'wear', 'assemble', 'pull'],
                        {'w': 7.5, 'h': 2.5, 'a': 0, 'c': 'tab:purple', 'ox': 1., 'oy': -0.1, 'tox': -1.5, 'toy': -2.5}),
                  }
    elif exp == 'sl':
        sys.argv[1:] = ['--save_dir', 'output/skzs/hico_zsk_gc_nobg_sl/asl1/2019-09-25_10-25-31_SINGLE']
        offsets = {
            1: [0, 0],  # adjust
            2: [-1, -0.8],  # assemble
            3: [-0.5, -0.9],  # block
            4: [-0.7, 0.2],  # blow
            5: [0, -0.4],  # board
            6: [-1, -0.8],  # break
            7: [-1.6, 0.3],  # brush_with
            8: [-1, -0.3],  # buy
            9: [-1.05, 0.15],  # carry
            11: [-1, 0.1],  # chase
            12: [-0.9, 0.2],  # check
            13: [-0.4, 0.2],  # clean
            16: [0, -0.4],  # cut
            17: [-1, 0.4],  # cut_with
            18: [-1.1, -0.8],  # direct
            20: [0, -0.6],  # dribble
            21: [-1.5, -0.8],  # drink_with
            22: [0, -0.4],  # drive
            25: [-1, -0.8],  # eat_at
            27: [-1, 0],  # feed
            28: [0, 0],  # fill
            29: [-0.8, 0],  # flip
            30: [-0.6, 0.2],  # flush
            32: [-1.4, -0.4],  # greet
            33: [0.1, -0.2],  # grind
            34: [0.1, -0.4],  # groom
            35: [0, -0.7],  # herd
            36: [0, -0.4],  # hit
            38: [-1.2, 0.3],  # hop_on
            40: [-0.7, -0.8],  # hug
            41: [0, 0],  # hunt
            43: [0, 0],  # install
            45: [0, -0.6],  # kick
            46: [-0.7, 0.1],  # kiss
            48: [-0.9, 0.2],  # launch
            49: [0, 0],  # lick
            52: [0, -0.7],  # light
            53: [0, 0],  # load
            54: [0, -0.6],  # lose
            55: [-1, -0.8],  # make
            56: [0, -0.6],  # milk
            59: [-0.7, -0.8],  # operate
            60: [-1, -0.8],  # pack
            63: [0, -0.4],  # pay
            65: [0, -0.7],  # pet
            66: [0, -0.5],  # pick
            67: [0, -0.5],  # pick_up
            68: [-0.5, 0.4],  # point
            69: [-0.8, 0.3],  # pour
            70: [0.1, -0.4],  # pull
            73: [-1, 0],  # read
            74: [-1.2, -0.7],  # release
            75: [-0.7, 0.2],  # repair
            78: [0, 0],  # run
            79: [0, -0.6],  # sail
            80: [0, -0.3],  # scratch
            81: [-0.9, -0.8],  # serve
            82: [-0.9, 0.2],  # set
            84: [-0.3, -0.8],  # sign
            85: [0, -0.4],  # sip
            89: [-1.4, -0.5],  # smell
            90: [0, -0.6],  # spin
            91: [0, -0.3],  # squeeze
            93: [0, -0.2],  # stand_on
            94: [-1.3, 0.3],  # stand_under
            95: [-0.5, -0.8],  # stick
            97: [-1.6, -0.9],  # stop_at
            98: [0, -0.3],  # straddle
            102: [0, -0.4],  # teach
            103: [-1.2, -0.8],  # text_on
            104: [-0.8, -0.8],  # throw
            106: [-0.6, 0.3],  # toast
            107: [0, -0.7],  # train
            111: [0, -0.5],  # wash
            112: [0, -0.6],  # watch
            115: [-0.5, 0.3],  # wield
        }
        groups = {'A': (['sign', 'throw', 'hit', 'catch', 'spin', 'block', 'kick', 'serve', 'dribble'],
                        {'w': 6.0, 'h': 5.2, 'a': 60, 'c': 'tab:orange', 'ox': 0.5, 'oy': -0.2, 'tox': -2.3, 'toy': -3.3}),
                  'B': (['kiss', 'hug', 'pet', 'feed', 'watch'],
                        {'w': 5, 'h': 2, 'a': 75, 'c': 'tab:purple', 'ox': -0.5, 'oy': -0.5, 'tox': 0.2, 'toy': -3}),
                  'C': (['toast', 'fill', 'drink_with', 'pour', 'sip', 'lick', 'stir'],
                        {'w': 5.2, 'h': 4., 'a': -45, 'c': 'tab:brown', 'ox': 0.3, 'oy': 0.2, 'tox': -2, 'toy': -2.2}),
                  }
    elif exp == 'sl_ext':
        sys.argv[1:] = ['--save_dir', 'output/skzs/hico_zsk_gc_nobg_sl_ext2/wemboo_asl1/2019-10-23_13-52-02_RUN1']
        groups = {'A': (['launch', 'blow', 'hit', 'catch', 'release', 'kick', 'dribble'],
                        {'w': 5.5, 'h': 3, 'a': 0, 'c': 'tab:orange', 'ox': 0.1, 'oy': 0, 'tox': 1.0, 'toy': -2.2}),
                  'B': (['hunt', 'chase', 'pet', 'kiss', 'hug', 'scratch'],
                        {'w': 5, 'h': 3, 'a': 0, 'c': 'tab:purple', 'ox': 0.3, 'oy': 0.1, 'tox': 1.0, 'toy': -2.2}),
                  'C': (['drink_with', 'pour', 'sip', 'fill', 'stir'],
                        {'w': 4., 'h': 3, 'a': -45, 'c': 'tab:brown', 'ox': 0.2, 'oy': -0.1, 'tox': 2.0, 'toy': -1.5}),
                  }
        offsets = {
            2: [0, -0.6],  # assemble
            3: [-0.7, -0.6],  # block
            4: [0, -0.4],  # blow
            8: [-0.6, 0.1],  # buy
            9: [-0.6, -0.6],  # carry
            10: [-1.2, -0.6],  # catch
            21: [-0.3, -0.6],  # drink_with
            33: [0, -0.6],  # grind
            43: [0, -0.6],  # install
            45: [0, -0.4],  # kick
            48: [-1.2, -0.6],  # launch
            49: [-0.7, -0.6],  # lick
            54: [0, -0.6],  # lose
            69: [-0.9, 0],  # pour
            71: [-0.7, 0.1],  # push
            72: [-0.5, 0.1],  # race
            85: [0, -0.5],  # sip
            89: [0, -0.6],  # smell
            99: [0, -0.6],  # swing
            102: [0, -0.5],  # teach
            105: [0, -0.6],  # tie
            110: [0, -0.5],  # walk
        }
    elif exp == 'sl_ra':
        sys.argv[1:] = ['--save_dir', 'output/skzs/hico_zsk_gc_nobg_sl_Ra/wemboo_asl1_Ra-10-03_grseen/2019-10-24_11-59-36_RUN1']
        offsets = {
            1: [-1.3, 0],  # adjust
            3: [-0.8, 0.3],  # block
            5: [0, -0.6],  # board
            6: [-1, -0.8],  # break
            7: [-1.6, 0.3],  # brush_with
            8: [-1, -0.3],  # buy
            9: [-1.05, 0.15],  # carry
            11: [-1, 0.1],  # chase
            12: [-0.9, 0.2],  # check
            16: [0, -0.4],  # cut
            18: [-1.1, -0.8],  # direct
            20: [0, -0.6],  # dribble
            21: [0, 0],  # drink_with
            22: [-0.9, 0.25],  # drive
            25: [-1, -0.8],  # eat_at
            27: [-0.9, 0.1],  # feed
            28: [0, 0],  # fill
            29: [-0.8, 0],  # flip
            32: [-1.3, -0.4],  # greet
            33: [-0.9, -0.8],  # grind
            35: [0, -0.7],  # herd
            38: [0, -0.2],  # hop_on
            41: [-1, -0.5],  # hunt
            43: [-1.4, -0.4],  # install
            46: [-0.9, 0.1],  # kiss
            48: [-1.3, -0.8],  # launch
            49: [0, 0],  # lick
            52: [-1, -0.5],  # light
            53: [-1, -0.7],  # load
            54: [0, -0.6],  # lose
            55: [-1, 0.2],  # make
            56: [0, -0.6],  # milk
            59: [0, -0.4],  # operate
            60: [-1, -0.8],  # pack
            63: [0, -0.4],  # pay
            66: [-1, -0.3],  # pick
            67: [0, -0.5],  # pick_up
            69: [0, -0.4],  # pour
            73: [-1, 0],  # read
            74: [-1.2, -0.7],  # release
            78: [-0.7, 0],  # run
            79: [0, -0.6],  # sail
            80: [0, -0.3],  # scratch
            81: [-0.4, -0.8],  # serve
            85: [0, -0.4],  # sip
            89: [-1, 0.1],  # smell
            90: [0, -0.6],  # spin
            91: [0, -0.3],  # squeeze
            93: [0, -0.2],  # stand_on
            94: [-1.3, 0.3],  # stand_under
            95: [0, -0.3],  # stick
            97: [-0.4, -0.8],  # stop_at
            98: [0, -0.3],  # straddle
            102: [0, -0.4],  # teach
            103: [-1.2, -0.8],  # text_on
            106: [0, 0],  # toast
            107: [0, -0.3],  # train
            111: [0, -0.5],  # wash
            112: [0.1, -0.4],  # watch
        }
        groups = {'A': (['sign', 'throw', 'hit', 'catch', 'spin', 'block', 'kick', 'serve', 'dribble'],
                        {'w': 5.8, 'h': 4.5, 'a': 45, 'c': 'tab:orange', 'ox': 0.5, 'oy': 0, 'tox': -2.2, 'toy': -3.2}),
                  'B': (['kiss', 'hug', 'pet', 'feed', 'watch'],
                        {'w': 6, 'h': 2.5, 'a': 60, 'c': 'tab:purple', 'ox': 0.25, 'oy': 0.15, 'tox': -1.5, 'toy': 1}),
                  'C': (['toast', 'fill', 'drink_with', 'pour', 'sip'],
                        {'w': 5.5, 'h': 3.5, 'a': 0, 'c': 'tab:brown', 'ox': 0.9, 'oy': 0.3, 'tox': 3, 'toy': -2.2}),
                  }
    elif exp == 'sl_ra_ext':
        sys.argv[1:] = ['--save_dir', 'output/skzs/hico_zsk_gc_nobg_sl_Ra_ext2/wemboo_asl1_Ra-10-03_grseen/2019-10-24_13-32-02_RUN2']
        # perplexity = 40.0
    elif exp == 'glove':
        pass
    else:
        raise NotImplementedError

    dataset = Hico()
    n = 10
    print(' ' * 3, end='  ')
    for i in range(n):
        print(f'{i:<20d}', end=' ')
    for i, a in enumerate(dataset.actions):
        if i % n == 0:
            print()
            print(f'{i // n:3d}', end=': ')
        print(f'{a:20s}', end=' ')
    print()

    if exp != 'glove':
        print(sys.argv)
        cfg.parse_args(fail_if_missing=False)
        cfg.load()
        inds_dict = pickle.load(open(cfg.seen_classes_file, 'rb'))
    else:
        inds_dict = pickle.load(open('zero-shot_inds/seen_inds_0.pkl.push', 'rb'))

    seen_act_inds = np.array(sorted(inds_dict[Splits.TRAIN.value]['act'].tolist()))
    unseen_act_inds = np.setdiff1d(np.arange(dataset.num_actions), seen_act_inds)
    seen_act_inds = np.setdiff1d(seen_act_inds, np.array([0]))  # remove null

    if exp == 'glove':
        from lib.dataset.word_embeddings import WordEmbeddings
        word_embs = WordEmbeddings(source='glove', dim=300, normalize=True)
        act_class_embs = word_embs.get_embeddings(dataset.actions, retry='avg')
    else:
        act_class_embs = np.load(os.path.join(cfg.output_analysis_path, 'act_embs.npy'))
    act_emb_2d = TSNE(perplexity=perplexity).fit_transform(act_class_embs)

    fig, ax = plt.subplots(figsize=(14.4, 9))
    for ainds, c in [(seen_act_inds, 'tab:green'),
                     (unseen_act_inds, 'tab:red')]:
        x, y = act_emb_2d[ainds, 0], act_emb_2d[ainds, 1]
        ax.scatter(x, y, c=c)
        for i, act_i in enumerate(ainds):
            assert act_i > 0
            txt = dataset.actions[act_i] if act_i > 0 else 'NULL'
            pos = np.array([x[i], y[i]]) + np.array(offsets.get(act_i, [0, 0]))
            ax.annotate(txt, xy=(x[i], y[i]), xytext=pos + np.array([0.1, 0.1]), fontsize=12)
    # ax.set_title(f'Perplexity = {perplexity}')
    ax.axis('off')
    print(f'Perplexity = {perplexity}')

    for k, (g, p) in groups.items():
        inds = np.array([dataset.action_index[a] for a in g])
        x, y = act_emb_2d[inds, 0], act_emb_2d[inds, 1]

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        ellipse = Ellipse(xy=(mean_x + p['ox'], mean_y + p['oy']), width=p['w'], height=p['h'], angle=p['a'],
                          facecolor='none', edgecolor=p['c'], linewidth=2)
        ax.add_patch(ellipse)
        ax.annotate(k, xy=(mean_x + p['tox'], mean_y + p['toy']), fontsize=24, color=p['c'], fontweight='bold')

    plt.savefig(fig_fn, dpi=400, bbox_inches='tight')

    # mng = plt.get_current_fig_manager()
    # try:
    #     mng.frame.Maximize(True)
    # except AttributeError:
    #     mng.window.showMaximized()
    plt.show()


if __name__ == '__main__':
    # if torch.cuda.is_available():
    #     save()
    # else:
    show()
