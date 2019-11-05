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
        inds_dict = pickle.load(open(cfg.active_classes_file, 'rb'))
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
    # sys.argv[1:] = ['--save_dir', 'output/skzs/hico_zsk_gc_nobg_sl/asl1/2019-09-25_10-25-31_SINGLE']
    # sys.argv[1:] = ['--save_dir', 'output/skzs/hico_zsk_gc_nobg_Ra/Ra-10-03/2019-09-25_10-25-51_SINGLE/']
    # sys.argv[1:] = ['--save_dir', 'output/skzs/hico_zsk_gc_nobg_sl_Ra/asl1_Ra-10-03/2019-09-25_14-21-33_SINGLE']
    sys.argv[1:] = ['--save_dir', 'output/skzs/hico_zsk_gc_nobg_sl_Ra/wemboo_asl1_Ra-10-03_grseen/2019-10-24_11-59-36_RUN1']
    print(sys.argv)
    cfg.parse_args(fail_if_missing=False)
    cfg.load()

    np.random.seed(3)

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

    inds_dict = pickle.load(open(cfg.active_classes_file, 'rb'))
    seen_act_inds = np.array(sorted(inds_dict[Splits.TRAIN.value]['act'].tolist()))
    unseen_act_inds = np.setdiff1d(np.arange(dataset.num_actions), seen_act_inds)

    act_class_embs = np.load(os.path.join(cfg.output_analysis_path, 'act_embs.npy'))
    perplexity = 20.0
    act_emb_2d = TSNE(perplexity=perplexity).fit_transform(act_class_embs)

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
        27: [-1, 0],  # feed
        28: [0, 0],  # fill
        29: [-0.8, 0],  # flip
        32: [-1.3, -0.4],  # greet
        33: [-0.9, -0.8],  # grind
        35: [0, -0.7],  # herd
        38: [0, -0.2],  # hop_on
        41: [-1, -0.5],  # hunt
        43: [-1.4, -0.4],  # install
        46: [-0.9, 0],  # kiss
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
        112: [0, -0.6],  # watch
    }
    fig, ax = plt.subplots(figsize=(14.4, 9))
    for ainds, c in [(seen_act_inds, 'tab:green'),
                     (unseen_act_inds, 'tab:red')]:
        x, y = act_emb_2d[ainds, 0], act_emb_2d[ainds, 1]
        ax.scatter(x, y, c=c)
        for i, act_i in enumerate(ainds):
            txt = dataset.actions[act_i] if act_i > 0 else 'NULL'
            pos = np.array([x[i], y[i]]) + np.array(offsets.get(act_i, [0, 0]))
            ax.annotate(txt, xy=(x[i], y[i]), xytext=pos + np.array([0.1, 0.1]), fontsize=12)
    # ax.set_title(f'Perplexity = {perplexity}')
    ax.axis('off')
    print(f'Perplexity = {perplexity}')

    groups = {'A': (['sign', 'throw', 'hit', 'catch', 'spin', 'block', 'kick', 'serve', 'dribble'],
                    {'w': 5.8, 'h': 4.5, 'a': 45, 'c': 'tab:orange', 'ox': 0.5, 'oy': 0, 'tox': -2.2, 'toy': -3.2}),
              'B': (['shear', 'kiss', 'hug', 'pet', 'feed', 'watch'],
                    {'w': 7, 'h': 3, 'a': 75, 'c': 'tab:purple', 'ox': 0.15, 'oy': 0.25, 'tox': -1.5, 'toy': 2}),
              'C': (['toast', 'fill', 'drink_with', 'pour', 'sip', 'lick'],
                    {'w': 6.5, 'h': 4.2, 'a': 10, 'c': 'tab:brown', 'ox': 0.7, 'oy': 0, 'tox': 3, 'toy': -2.2}),
              }
    # plt.savefig('/home/alex/Dropbox/PhD Docs/My stuff/Conferences/Second paper/v5.1 - CVPR/images/tsne_sl_ra_text.png', dpi=400,
    # bbox_inches='tight')

    for k, (g, p) in groups.items():
        inds = np.array([dataset.action_index[a] for a in g])
        x, y = act_emb_2d[inds, 0], act_emb_2d[inds, 1]

        mean_x = np.mean(x)
        mean_y = np.mean(y)
        transf = transforms.Affine2D().rotate_deg(p['a']).translate(mean_x + p['ox'], mean_y + p['oy'])
        ellipse = Ellipse((0, 0), width=p['w'], height=p['h'], facecolor='none', edgecolor=p['c'], linewidth=2)
        ellipse.set_transform(transf + ax.transData)
        ax.add_patch(ellipse)
        ax.annotate(k, xy=(mean_x + p['tox'], mean_y + p['toy']), fontsize=24, color=p['c'], fontweight='bold')
    plt.savefig('/home/alex/Dropbox/PhD Docs/My stuff/Conferences/Second paper/v5.1 - CVPR/images/tsne_sl_ra_text_marked.png', dpi=400,
                bbox_inches='tight')

    # mng = plt.get_current_fig_manager()
    # try:
    #     mng.frame.Maximize(True)
    # except AttributeError:
    #     mng.window.showMaximized()
    plt.show()


if __name__ == '__main__':
    if torch.cuda.is_available():
        save()
    else:
        show()
