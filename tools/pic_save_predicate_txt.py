import os
import json


def convert_to_txt():
    for src_file in [LABEL_CATS_FILE, RELATION_CATS_FILE]:
        with open(os.path.join(DATA_DIR, src_file), 'r', encoding='utf-8') as f:
            src_cats = json.load(f)

        categories = []
        for cat_dict in src_cats:
            categories.append(cat_dict['name'])

        for i, c in enumerate(categories):
            assert i == src_cats[i]['id']

        print('\n'.join(categories))

        with open(os.path.join(DATA_DIR, src_file.replace('json', 'txt')), 'w', encoding='utf-8') as f:
            f.writelines('\n'.join(categories))
        print()
        print('=' * 50)
        print()


def main():
    pass


if __name__ == '__main__':
    DATA_DIR = os.path.join('data', 'PiC')
    LABEL_CATS_FILE = 'label_categories.json'
    RELATION_CATS_FILE = 'relation_categories.json'
    convert_to_txt()
    main()
