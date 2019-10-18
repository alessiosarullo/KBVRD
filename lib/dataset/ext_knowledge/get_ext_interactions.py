import json
import os
import pickle

import numpy as np

from config import cfg
from lib.dataset.ext_knowledge.ext_source import HCVRD, VG, ActivityNetCaptions
from lib.dataset.ext_knowledge.imsitu import ImSitu
from lib.dataset.hoi_dataset import HoiDataset

# FIXME the whole thing is coded terribly (dataset classes as well)

person_words = {'person', 'man', 'woman', 'boy', 'girl', 'child', 'kid', 'baby', 'guy',
                'audience', 'catcher', 'carrier', 'classroom', 'couple', 'cowboy', 'crowd', 'driver', 'friend', 'guard', 'little girl',
                'player', 'rider', 'skateboarder', 'skater', 'skier', 'small child', 'snowboarder', 'surfer', 'tennis player'}


def parse_captions(captions, hoi_ds: HoiDataset):
    from nltk.corpus import wordnet as wn
    from nltk.tag import pos_tag
    from nltk.tokenize import word_tokenize

    # If not downloaded already, run:
    # nltk.download('punkt')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('universal_tagset')
    # nltk.download('wordnet')

    # stop_words = stopwords.words('english') + list(get_stop_words('en'))

    action_verbs = [hoi_ds.actions[0]] + [p.split('_')[0] for p in hoi_ds.actions[1:]]
    action_set = set(action_verbs)
    objs_per_action = {p: set() for p in hoi_ds.actions}
    for i_cap, caption in enumerate(captions):
        if i_cap % 1000 == 0:
            print(i_cap, '/', len(captions))

        tokens = word_tokenize(caption)
        tagged_tokens = pos_tag(tokens, tagset='universal')

        # Find person and verb
        person_found = False
        for i_tokens, w in enumerate(tokens):
            if wn.morphy(w, wn.NOUN) in person_words:
                person_found = True
            else:
                verb = wn.morphy(w, wn.VERB)
                if verb in action_set:
                    break
        else:
            continue

        # Either no human or no object. Skip this iteration.
        if not person_found or i_tokens + 1 == len(tokens):
            continue

        # If there is a preposition, it will be counted as part of the predicate and thus removed from the words that follow.
        following_tagged_tokens = tagged_tokens[i_tokens + 1:]
        if following_tagged_tokens[0][1] == 'ADP':
            preposition = following_tagged_tokens[0][0]
            following_tagged_tokens = following_tagged_tokens[1:]
        else:
            preposition = None

        # Find object phrase. This basically keeps going as long as it finds adjectives and similar, until it finds either nouns (success) or
        # something else such as verbs (failure).
        p_obj_sentence = []
        for w, pos in following_tagged_tokens:
            if pos == 'NOUN' and w not in ['front', 'top']:
                p_obj_sentence.append(w)
            elif pos in {'NOUN', 'PRON', 'ADJ', 'DET'}:
                continue
            else:
                break
        if not p_obj_sentence:
            continue
        else:
            p_obj = ' '.join(p_obj_sentence)

        for i_pred, orig_a in enumerate(hoi_ds.actions):
            if action_verbs[i_pred] == verb:
                p_tokens = orig_a.split('_')
                if len(p_tokens) > 1 and (preposition is None or preposition != p_tokens[1]):
                    continue
                objs_per_action[orig_a].add(p_obj)

    for p, objs in objs_per_action.items():
        print('%20s:' % p, sorted(objs))

    return objs_per_action


def get_interactions_from_vg(hoi_ds: HoiDataset):
    data_dir = os.path.join(cfg.data_root, 'VG')
    try:
        with open(os.path.join(cfg.cache_root, 'vg_action_objects.pkl'), 'rb') as f:
            objs_per_actions = pickle.load(f)
    except FileNotFoundError:
        try:
            with open(os.path.join(cfg.cache_root, 'vg_region_descriptions.txt'), 'r') as f:
                region_descr = [l.strip() for l in f.readlines()]
        except FileNotFoundError:
            region_descr = json.load(open(os.path.join(data_dir, 'region_descriptions.json'), 'r'))
            region_descr = [r['phrase'] for rd in region_descr for r in rd['regions']]
            with open(os.path.join(cfg.cache_root, 'vg_region_descriptions.txt'), 'w') as f:
                f.write('\n'.join(region_descr))
        print('\n'.join(region_descr[:10]))
        print()

        objs_per_actions = parse_captions(region_descr, hoi_ds)
        with open(os.path.join(cfg.cache_root, 'vg_action_objects.pkl'), 'wb') as f:
            pickle.dump(objs_per_actions, f)
        print(len(region_descr))

    interactions = np.array([[hoi_ds.action_index.get(a, -1), hoi_ds.object_index.get(o, -1)] for a, objs in objs_per_actions.items() for o in objs])
    interactions = np.unique(interactions[np.all(interactions >= 0, axis=1), :], axis=0)
    return interactions


def get_interactions_from_vcap(hoi_ds: HoiDataset):
    try:
        with open(os.path.join(cfg.cache_root, 'vcap_predicate_objects.pkl'), 'rb') as f:
            objs_per_actions = pickle.load(f)
    except FileNotFoundError:
        d = json.load(open(os.path.join(cfg.data_root, 'VideoCaptions', 'train.json'), 'r'))
        captions = [s.strip(' .') for v in d.values() for s in v['sentences']]
        print('\n'.join(captions[:10]))
        print()

        objs_per_actions = parse_captions(captions, hoi_ds)
        with open(os.path.join(cfg.cache_root, 'vcap_predicate_objects.pkl'), 'wb') as f:
            pickle.dump(objs_per_actions, f)
    interactions = np.array([[hoi_ds.action_index.get(a, -1), hoi_ds.object_index.get(o, -1)] for a, objs in objs_per_actions.items() for o in objs])
    interactions = interactions[np.all(interactions >= 0, axis=1), :]
    return interactions


def get_interactions_from_imsitu(hoi_ds: HoiDataset, imsitu: ImSitu):
    op_mat = imsitu.extract_freq_matrix(hoi_ds)
    o, p = np.where(op_mat > 0)
    interactions = np.stack([p, o], axis=1)
    return interactions


def get_seen_interactions(hoi_ds: HoiDataset, seenf=0):
    cfg.seenf = seenf
    inds_dict = pickle.load(open(cfg.active_classes_file, 'rb'))
    obj_inds = sorted(inds_dict['train']['obj'].tolist())
    act_inds = sorted(inds_dict['train']['act'].tolist())
    interactions_inds = np.setdiff1d(np.unique(hoi_ds.oa_pair_to_interaction[obj_inds, :][:, act_inds]), np.array([-1]))
    interactions = hoi_ds.interactions[interactions_inds, :]
    return interactions


def get_uncovered_interactions(hoi_ds_interactions, *ext_interactions, include_null=False):
    hoi_ds_set = {tuple(x) for x in hoi_ds_interactions}
    ext_set = {tuple(x) for e_inters in ext_interactions for x in e_inters}
    uncovered_interactions = np.array(sorted([x for x in hoi_ds_set - ext_set]))
    if not include_null:
        uncovered_interactions = uncovered_interactions[uncovered_interactions[:, 0] > 0, :]
    return uncovered_interactions


def compute_isolated(all_interactions, uncovered_interactions, idx, num_classes):
    ids, num_links = np.unique(all_interactions[:, idx], return_counts=True)
    assert np.all(ids == np.arange(num_classes))
    for x in uncovered_interactions[:, idx]:
        num_links[x] -= 1
    assert np.all(num_links >= 0)
    isolated = np.flatnonzero(num_links == 0)
    return isolated


def get_interactions_from_ext_src(hoi_ds: HoiDataset):
    hcvrd = HCVRD()
    imsitu = ImSitu()
    vg_interactions = get_interactions_from_vg(hoi_ds)
    vcap_interactions = get_interactions_from_vcap(hoi_ds)
    hcvrd_interactions = hcvrd.get_interactions_for(hoi_ds)
    imsitu_interactions = get_interactions_from_imsitu(hoi_ds, imsitu)
    ext_interactions = np.concatenate([vg_interactions, vcap_interactions, hcvrd_interactions, imsitu_interactions], axis=0)
    return ext_interactions


def check():
    from lib.dataset.hico import Hico
    hico = Hico()

    train_interactions = get_seen_interactions(hico, seenf=0)
    print('%15s' % 'Train', get_uncovered_interactions(hico.interactions, train_interactions).shape[0])

    vg_interactions = get_interactions_from_vg(hico)
    print('%15s' % 'VG', get_uncovered_interactions(hico.interactions, vg_interactions).shape[0])
    print('%15s' % 'VG-train', get_uncovered_interactions(hico.interactions, train_interactions, vg_interactions).shape[0])

    vcap_interactions = get_interactions_from_vcap(hico)
    print('%15s' % 'VCAP', get_uncovered_interactions(hico.interactions, vcap_interactions).shape[0])
    print('%15s' % 'VCAP-train', get_uncovered_interactions(hico.interactions, train_interactions, vcap_interactions).shape[0])

    hcvrd = HCVRD()
    hcvrd_interactions = hcvrd.get_interactions_for(hico)
    print('%15s' % 'HCVRD', get_uncovered_interactions(hico.interactions, hcvrd_interactions).shape[0])
    print('%15s' % 'HCVRD-train', get_uncovered_interactions(hico.interactions, train_interactions, hcvrd_interactions).shape[0])

    imsitu = ImSitu()
    imsitu_interactions = get_interactions_from_imsitu(hico, imsitu)
    print('%15s' % 'ImSitu', get_uncovered_interactions(hico.interactions, imsitu_interactions).shape[0])
    print('%15s' % 'ImSitu-train', get_uncovered_interactions(hico.interactions, train_interactions, imsitu_interactions).shape[0])

    uncovered_interactions = get_uncovered_interactions(hico.interactions, train_interactions, vg_interactions, hcvrd_interactions,
                                                        imsitu_interactions, vcap_interactions)
    print(uncovered_interactions.shape[0])

    isolated_actions = compute_isolated(hico.interactions, uncovered_interactions, idx=0, num_classes=hico.num_actions)
    print('Isolated actions:', [hico.actions[a] for a in isolated_actions])
    # ['hop_on', 'hunt', 'lose', 'pay', 'point', 'sign', 'stab', 'toast'].
    # 'hop_on' and 'sign' (and maybe 'point') could probably be found through synonyms. The others are too niche/hard to find (hunt, stab, lose)
    # or even borderline incorrect ("toast wine glass").

    isolated_objects = compute_isolated(hico.interactions, uncovered_interactions, idx=1, num_classes=hico.num_objects)
    print('Isolated objects:', [hico.objects[o] for o in isolated_objects])


def main():
    from lib.dataset.hico import Hico
    hico = Hico()

    # hcvrd = HCVRD()
    # hcvrd.get_interactions_for(hico)

    # vg_interactions_old = get_interactions_from_vg(hico)
    # vgcaptions = VGCaptions(required_words={o.split('_')[-1] for o in hico.objects} | {a.split('_')[0] for a in hico.actions[1:]})
    # vg_interactions = vgcaptions.get_interactions_for(hico)
    # sep = '###'
    # vgo = {f'{a}{sep}{o}' for a, o in vg_interactions_old}
    # vg = {f'{a}{sep}{o}' for a, o in vg_interactions}
    # print(len(vg - vgo), len(vgo - vg))
    # print()

    # vg = VG()
    # vg_interactions = vg.get_interactions_for(hico)
    # vg_interactions_old = get_interactions_from_vg(hico)
    # sep = '###'
    # vgo = {f'{a}{sep}{o}' for a, o in vg_interactions_old}
    # vg = {f'{a}{sep}{o}' for a, o in vg_interactions}
    # print(len(vg - vgo), len(vgo - vg))
    # print()

    anet_interactions_old = get_interactions_from_vcap(hico)
    anet = ActivityNetCaptions(required_words={o.split('_')[-1] for o in hico.objects} | {a.split('_')[0] for a in hico.actions[1:]})
    anet_interactions = anet.get_interactions_for(hico)
    sep = '###'
    old = {f'{a}{sep}{o}' for a, o in anet_interactions_old}
    new = {f'{a}{sep}{o}' for a, o in anet_interactions}
    print(len(new - old), len(old - new))
    print()


if __name__ == '__main__':
    main()
    # check()
