"""
Adapted from PyTorch's text library.
"""

import array
import os

import six
import torch

from config import cfg

# TODO merge with WordEmbeddings

# FIXME check and update
def obj_edge_vectors(names, wv_dim=300) -> torch.Tensor:
    wv_dict, wv_arr, wv_size = load_word_vectors(wv_type='glove.6B', dim=wv_dim)

    vectors = torch.Tensor(len(names), wv_dim)
    vectors.normal_(0, 1)

    for i, token in enumerate(names):
        wv_index = wv_dict.get(token, None)
        if wv_index is not None:
            vectors[i] = wv_arr[wv_index]
        else:
            # Try the longest word (hopefully won't be a preposition
            lw_token = sorted(token.replace('_', ' ').strip().split(' '), key=lambda x: len(x), reverse=True)[0]
            print("{} -> {} ".format(token, lw_token))
            wv_index = wv_dict.get(lw_token, None)
            if wv_index is not None:
                vectors[i] = wv_arr[wv_index]
            else:
                print("fail on {}".format(token))

    return vectors


def load_word_vectors(wv_type, dim):
    root = cfg.program.embedding_dir
    """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
    if isinstance(dim, int):
        dim = str(dim) + 'd'
    fname = os.path.join(root, wv_type + '.' + dim)
    if os.path.isfile(fname + '.pt'):
        fname_pt = fname + '.pt'
        print('Loading word vectors from', fname_pt)
        return torch.load(fname_pt)
    if os.path.isfile(fname + '.txt'):
        fname_txt = fname + '.txt'
        cm = open(fname_txt, 'rb')
        cm = [line for line in cm]
    else:
        raise RuntimeError('Unable to load word vectors from %s' % fname)

    wv_tokens, wv_arr, wv_size = [], array.array('d'), None
    if cm is not None:
        for line in range(len(cm)):
            entries = cm[line].strip().split(b' ')
            word, entries = entries[0], entries[1:]
            if wv_size is None:
                wv_size = len(entries)
            try:
                if isinstance(word, six.binary_type):
                    word = word.decode('utf-8')
            except:
                print('non-UTF8 token', repr(word), 'ignored')
                continue
            wv_arr.extend(float(x) for x in entries)
            wv_tokens.append(word)

    wv_dict = {word: i for i, word in enumerate(wv_tokens)}
    wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
    ret = (wv_dict, wv_arr, wv_size)
    torch.save(ret, fname + '.pt')
    return ret
