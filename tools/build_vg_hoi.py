import json
import pickle
import os


def main():
    relationships_data = json.load(open(os.path.join('data', 'VG', 'relationships.json'), 'r'))

    image_ids = [img_rdata['image_id'] for img_rdata in relationships_data]
    relationships = []
    for img_rdata in relationships_data:
        img_rs = []
        for rdict in img_rdata['relationships']:
            try:
                subj = rdict['subject']['name']
            except KeyError:
                subj = rdict['subject']['names'][0]
            pred = rdict['predicate']
            try:
                obj = rdict['object']['name']
            except KeyError:
                obj = rdict['object']['names'][0]
            img_rs.append([subj, pred, obj])
        relationships.append(img_rs)

    with open('vg_hoi.pkl', 'wb') as f:
        pickle.dump({'image_ids': image_ids,
                     'relationships': relationships},
                    f)


if __name__ == '__main__':
    main()
