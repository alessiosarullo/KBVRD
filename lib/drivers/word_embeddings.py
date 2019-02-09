import os
import pickle
import re

import numpy as np


class WordEmbeddings:
    def __init__(self, source='numberbatch'):
        """
        Attributes:
            embeddings: [array] NxD matrix consisting of N D-dimensional embeddings
            vocabulary: [list] the N words composing the vocabulary, sorted according to `embeddings`'s rows
        """

        self.data_dir = os.path.join('data', 'embeddings')
        self.loaders = {'numberbatch': {'parser': self.parse_numberbatch,
                                        'src_file': 'numberbatch-en.txt',
                                        },
                        'glove': {'parser': self.parse_glove,
                                  'src_file': 'glove.6B.300d.txt',
                                  },
                        }

        self.normalize = True
        try:
            self._embeddings, self.vocabulary = self.load(source.lower())
        except KeyError:
            raise ValueError('Unknown source %s. Possible sources:' % source, list(self.loaders.keys()))

        self.word_index = {v: i for i, v in enumerate(self.vocabulary)}

    def embedding(self, word):
        try:
            if word == 'hair drier':  # FIXME hard coded
                word = 'hair dryer'
            return self._embeddings[self.word_index[word], :]
        except KeyError as e:
            return np.zeros_like(self._embeddings[0, :])  # FIXME
            # new_message = str(e).replace("'", '')
            # print(new_message)
            # if ' ' in word:
            #     new_message += '. Suggestions: [%s]' % ', '.join([w for w in self.word_index if w.startswith(word.split()[0])])
            # raise KeyError(new_message)

    def load(self, source):
        src_fn = self.loaders[source]['src_file']
        path_cache_file = os.path.join(self.data_dir, os.path.splitext(src_fn)[0] + '_cache.pkl')
        try:
            with open(path_cache_file, 'rb') as f:
                print('Loading cached %s embeddings' % source)
                embedding_mat, vocabulary = pickle.load(f)
        except FileNotFoundError:
            print('Parsing %s embeddings' % source)
            embedding_mat, vocabulary = self.loaders[source]['parser'](os.path.join(self.data_dir, src_fn))
            print('Cleaning')
            clean_words_inds = [i for i, word in enumerate(vocabulary) if not bool(re.search(r"[^a-zA-Z0-9_'\-]", word))]
            vocabulary = [vocabulary[i].replace('_', ' ') for i in clean_words_inds]
            embedding_mat = embedding_mat[clean_words_inds, :]
            if self.normalize:
                print('Normalizing')
                norms = np.linalg.norm(embedding_mat, axis=1)
                norms[norms == 0] = 1
                embedding_mat /= norms[:, None]
            else:
                print('Using unnormalized embeddings')
            with open(path_cache_file, 'wb') as f:
                print('Caching results')
                pickle.dump((embedding_mat, vocabulary), f)
        print('Done')
        return embedding_mat, vocabulary

    @staticmethod
    def parse_glove(src_file):
        with open(src_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]

        embeddings, vocabulary = [], []
        for i, line in enumerate(lines):
            tokens = line.split()
            embeddings.append(np.array([float(x) for x in tokens[1:]]))
            vocabulary.append(tokens[0])

        embedding_mat = np.stack(embeddings, axis=0)
        return embedding_mat, vocabulary

    @staticmethod
    def parse_numberbatch(src_file):
        """
        Format (from https://github.com/commonsense/conceptnet-numberbatch):
            The first line of the file contains the dimensions of the matrix:
                1984681 300
            Each line contains a term label followed by 300 floating-point numbers, separated by spaces:
                /c/en/absolute_value -0.0847 -0.1316 -0.0800 -0.0708 -0.2514 -0.1687 -...
                /c/en/absolute_zero 0.0056 -0.0051 0.0332 -0.1525 -0.0955 -0.0902 0.07...
                /c/en/absoluteless 0.2740 0.0718 0.1548 0.1118 -0.1669 -0.0216 -0.0508...
                /c/en/absolutely 0.0065 -0.1813 0.0335 0.0991 -0.1123 0.0060 -0.0009 0...
                /c/en/absolutely_convergent 0.3752 0.1087 -0.1299 -0.0796 -0.2753 -0.1...
        :return: see __init__
        """

        with open(src_file, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines()]

        embedding_mat = np.zeros([int(dim) for dim in lines[0].split()])
        vocabulary = []
        for i, line in enumerate(lines[1:]):
            tokens = line.split()
            embedding_mat[i, :] = np.array([float(x) for x in tokens[1:]])

            # # Words do not start with '/c/en/' in the english-only version
            # word_id_tokens = tokens[0].split('/')
            # assert len(word_id_tokens) == 4, (word_id_tokens, i, line)
            # vocabulary.append(word_id_tokens[-1])

            vocabulary.append(tokens[0])

        return embedding_mat, vocabulary


def main():
    we = WordEmbeddings(source='glove')
    print(we.vocabulary[:50])
    print(we._embeddings[:5, :5])


if __name__ == '__main__':
    main()
