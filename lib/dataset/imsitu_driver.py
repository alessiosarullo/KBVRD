import json
import os

from typing import Dict, Any, Tuple

from config import cfg

class ImSitu:
    def __init__(self):
        """
        Attributes:
            - nouns [dict(dict)]: More than 80k entries. Keys are ImageNet synsets, which are in turn derived from WordNet 3.0. Values are
                dictionaries containing the following items:
                    - 'gloss' [list(str)]: List of nouns  describing the concept.
                    - 'def' [str]: A definition.
                EXAMPLE: Key: 'n03024882'. Value:
                    {'gloss': ['choker', 'collar', 'dog collar', 'neckband'],
                     'def': "necklace that fits tightly around a woman's neck"
                     }
            - verbs [dict]: Around 500 entries. Keys are verbs themselves [str], while values are dictionaries of:
                - 'framenet' [str]: ID of the verb in FrameNet. Seems to somehow describe the category the verb belongs to.
                - 'def' [str]: Definition of the verb.
                - 'roles' [dict]: A dictionaries of the roles involved in the action specified by this verb. Keys vary according to the verb and
                    each item contains:
                        - 'framenet': See above.
                        - 'def' [str]: Describes the role the item specified by this key has.
                - 'abstract' [str]: A string describing the action on a general level.
                - 'order' [list(str)]: The order the roles appear in `abstract`.
                EXAMPLE: Key: 'tattoing'. Value:
                    {'framenet': 'Create_physical_artwork',
                     'def': 'to mark the skin with permanent colors and patterns',
                     'roles':
                        {'tool': {'framenet': 'instrument', 'def': 'The tool used'},
                         'place': {'framenet': 'place', 'def': 'The location where the tattoo event is happening'},
                         'target': {'framenet': 'representation', 'def': 'The entity being tattooed'},
                         'agent': {'framenet': 'creator', 'def': 'The entity doing the tattoo action'}
                         }
                     'abstract': 'AGENT tattooed TARGET with TOOL in PLACE',
                     'order': ['agent', 'target', 'tool', 'place'],
                    }
            - train, val, test [dict(dict)]: Keys are image file names, values are dictionaries with the following keys:
                - 'verb' [str]: Verb describing the image. It's a key for `verbs`.
                - 'frames' [list(dict)]: Each item is a dictionary. Keys are the roles specified in `verbs` for this verb, taking their values from
                    `nouns`'s keys.
                EXAMPLE: Key: 'glaring_215.jpg'. Value:
                    {'verb': 'glaring',
                     'frames': [{'place': 'n04215402', 'perceiver': '', 'agent': 'n10287213'},
                                {'place': 'n08613733', 'perceiver': '', 'agent': 'n10287213'},
                                {'place': 'n08613733', 'perceiver': '', 'agent': 'n10287213'}
                                ]
                    }
        """
        data_dir = os.path.join(cfg.program.data_root, 'imSitu')
        self.image_dir = os.path.join(data_dir, 'images')
        self.path_domain_file = os.path.join(data_dir, 'imsitu_space.json')
        self.path_train_file = os.path.join(data_dir, 'train.json')
        self.path_val_file = os.path.join(data_dir, 'dev.json')
        self.path_test_file = os.path.join(data_dir, 'test.json')

        self.nouns, self.verbs, self.train, self.val, self.test = self.load()
        self.verbs = self.fix_verbs(self.verbs)

    @staticmethod
    def fix_verbs(verbs) -> Dict[str, Dict]:
        verbs['riding']['abstract'] = verbs['riding']['abstract'].replace(' then ', ' the ')
        verbs['teaching']['abstract'] = verbs['teaching']['abstract'].replace(' to teach ', ' teaches ')
        return verbs

    def get_annotations(self, split):
        splits = {'train': self.train,
                  'val': self.val,
                  'test': self.test,
                  }
        return splits[split]

    def _print_verb_entry(self, verb):
        print('ImSitu verb entry example:')
        for k, v in self.verbs[verb].items():
            if k != 'roles':
                print('%15s: %s' % (k, v))
            else:
                print('%15s:' % k)
                ln = max([len(r) for r in v.keys()]) + 1
                for r, d in v.items():
                    print(('%15s  - %-' + str(ln) + 's %s') % ('', r + ':', d))

    def load(self) -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
        with open(self.path_domain_file, 'r') as f:
            domain = json.load(f)
        verbs, nouns = domain['verbs'], domain['nouns']

        with open(self.path_train_file, 'r') as f:
            train = json.load(f)
        with open(self.path_val_file, 'r') as f:
            val = json.load(f)
        with open(self.path_test_file, 'r') as f:
            test = json.load(f)

        return nouns, verbs, train, val, test


def main():
    imsitu = ImSitu()
    imsitu._print_verb_entry('riding')

if __name__ == '__main__':
    main()
