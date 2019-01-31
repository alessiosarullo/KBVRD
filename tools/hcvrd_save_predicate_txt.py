import os
import json


def save_predicates():
    with open(ANNOTATION_FILE, 'r', encoding='utf-8') as f:
        data_dict = json.load(f)

    predicates = set()
    for im_id, im_anns in data_dict.items():
        for ann in im_anns:
            try:
                predicates.add(ann['predicate'])
            except KeyError:
                print([k for k in ann])
                raise

    print(len(predicates))

    with open(os.path.join(DATA_DIR, 'predicates.txt'), 'w', encoding='utf-8') as f:
        f.writelines('\n'.join(sorted(list(predicates))))


def main():
    pass


if __name__ == '__main__':
    DATA_DIR = os.path.join('data', 'HCVRD')
    ANNOTATION_FILE = os.path.join(DATA_DIR, 'final_data.json')
    save_predicates()
    main()
