import json
import pickle
import os
from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag


def to_pickle():
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


def main():
    # %% VG HOI

    with open('vg_hoi.pkl', 'rb') as f:
        d = pickle.load(f)
        image_ids = d['image_ids']
        img_relationships = d['relationships']

    # %%

    relationships = [r for imgr in img_relationships for r in imgr]
    rel_img_ids = [image_ids[i] for i, imgr in enumerate(img_relationships) for r in imgr]
    relationships_inds = list(range(len(relationships)))

    # %%

    human_classes = {
        'people', 'men', 'classroom', 'couple', 'crowd', 'audience',
        'person', 'man', 'woman', 'boy', 'girl', 'child', 'kid', 'baby', 'guy', 'lady',
        'snowboarder', 'surfer', 'skier', 'player',
        'skateboarder', 'skater', 'rider', 'catcher',
        'carrier', 'cowboy', 'driver', 'friend', 'guard',
    }
    subjects = sorted({r[0].lower() for r in relationships if r[0]})
    print(len(subjects))
    print(len(human_classes))

    humans = {}
    for w in subjects:
        tokens = [t for t in w.split(' ') if t]
        tagged_tokens = pos_tag(tokens, tagset='universal')
        if not all([t.isalpha() for t in tokens]):
            continue
        if not all([tag in {'ADP', 'ADJ', 'NOUN', 'DET', 'NUM'} for tok, tag in tagged_tokens]):
            continue
        for t in tokens:
            m = t
            if m is not None and m in human_classes and w not in humans.keys():
                if len(tokens) > 1:
                    if m == 'baby':
                        break
                humans[w] = m
                break
    print(len(human_classes & humans.keys()))
    hp_set = set(humans.keys())
    print(len(hp_set))

    s_relationships_inds = [ri for ri in relationships_inds if relationships[ri][0] in hp_set]
    print(len(s_relationships_inds))
    hp_set = sorted(hp_set)

    # %%

    predicates = sorted({r[1].lower() for r in relationships if r[1]})
    print(len(predicates))

    FILTER_VERB_PHRASES = True

    blacklist = {'be', 'wear', 'have'}
    v_to_filter = {'back', 'arm',
                   'bottle', ' cap', 'bare',
                   'black', 'blue', 'brown', 'red', 'white', 'yellow', 'green'}
    t_to_filter = {'amused', 'amazed', 'colored'}
    verbs_phrases = {}
    for p in predicates:
        tokens = [t for t in p.split(' ') if t]
        if FILTER_VERB_PHRASES and len(tokens) > 1:
            continue
        t = tokens[0]
        verb = wn.morphy(t, wn.VERB)
        # if verb == 'be' and len(tokens) > 1:
        #     t = tokens[1]
        #     verb = wn.morphy(t, wn.VERB)
        if verb is not None and verb not in blacklist and \
                verb not in v_to_filter and t not in t_to_filter:
            is_vp = True
            if len(tokens) > 1:
                tagged_tokens = pos_tag(tokens[1:], tagset='universal')
                if not all([tag == 'ADP' for tok, tag in tagged_tokens]):
                    is_vp = False
            if is_vp:
                tokens[0] = verb
                vp = ' '.join(tokens)
                verbs_phrases[p] = vp
        else:
            vp = None
    vp_set = set(verbs_phrases.values())
    print(len(vp_set))

    p_relationships_inds = [ri for ri in relationships_inds if relationships[ri][1] in verbs_phrases.keys()]
    print(len(p_relationships_inds))

    # %%

    objects_orig = sorted({r[2].lower() for r in relationships if r[2]})
    print(len(objects_orig))

    FILTER_COMPOUND_NOUNS = True

    to_filter = {'an',
                 'umpire',
                 # 'head', 'leg', 'eye',
                 }
    objects = {}
    for w in objects_orig:
        tokens = [t for t in w.split(' ') if t]
        if FILTER_COMPOUND_NOUNS and len(tokens) > 1:
            continue
        tagged_tokens = pos_tag(tokens, tagset='universal')
        if not all([t.isalpha() for t in tokens]):
            continue
        if not all([tag in {'ADP', 'ADJ', 'NOUN', 'DET', 'NUM'} for tok, tag in tagged_tokens]):
            continue
        for t in tokens:
            m = wn.morphy(t, wn.NOUN)
            if m is not None and m not in to_filter:
                objects[w] = m
                break
    obj_set = set(objects.keys())
    print(len(obj_set))

    o_relationships_inds = [ri for ri in relationships_inds if relationships[ri][2] in obj_set]
    print(len(s_relationships_inds))
    obj_set = sorted(obj_set)

    # %%

    mapped_rels = [[humans.get(r[0], r[0]),
                    verbs_phrases.get(r[1], r[1]),
                    objects.get(r[2], r[2])]
                   for r in relationships]
    hoi_inds = sorted(set(s_relationships_inds) &
                      set(p_relationships_inds) &
                      set(o_relationships_inds))

    hoi_img_ids = [rel_img_ids[i] for i in hoi_inds]
    hois = [mapped_rels[i] for i in hoi_inds]
    print('Num imgs', len(set(hoi_img_ids)))
    print('Num nouns', len(set([hoi[i] for hoi in hois for i in [0, 2]])))
    print('Num objs', len(set([hoi[2] for hoi in hois])))
    print('Num verbs', len(set([hoi[1] for hoi in hois])))
    print('Num single-word verbs', len(set([hoi[1] for hoi in hois if len(hoi[1].split(' ')) == 1])))
    unique_hoi_str = sorted({f'{s} - {p} - {o}' for s, p, o in hois})
    unique_hoi_str_by_act = sorted(unique_hoi_str,
                                   key=lambda hoi_str: hoi_str.split(' - ')[1])
    print('Num HOIs', len(unique_hoi_str))

    # %%

    with open('tmp_act.txt', 'w') as f:
        f.write('\n'.join(sorted({p for s, p, o in hois})))

    # %%

    with open('tmp_hois_by_act.txt', 'w') as f:
        f.write('\n'.join([' '.join([f'{x:15s}' for x in s.split(' - ')]) for s in unique_hoi_str_by_act]))

    # %%

    for s, p, o in relationships:
        if 'come' in p:
            if 'slope' in o:
                print(f'{s}, {p}, {o}')


if __name__ == '__main__':
    # to_pickle()
    main()
