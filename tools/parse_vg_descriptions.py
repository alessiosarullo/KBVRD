import os
import pickle
import json

from nltk.corpus import wordnet as wn
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize

from lib.dataset.hicodet.hicodet import HicoDet
from config import cfg


def main():
    cfg.parse_args(fail_if_missing=False)

    hd = HicoDet()
    # stop_words = stopwords.words('english') + list(get_stop_words('en'))

    person_words = {'person', 'man', 'woman', 'boy', 'girl', 'child', 'kid', 'baby', 'guy',
                    'audience', 'catcher', 'carrier', 'classroom', 'couple', 'cowboy', 'crowd', 'driver', 'friend', 'guard', 'little girl',
                    'player', 'rider', 'skateboarder', 'skater', 'skier', 'small child', 'snowboarder', 'surfer', 'tennis player'}

    data_dir = os.path.join('data', 'VG')
    # rel_synsets = json.load(open(os.path.join(data_dir, 'relationship_synsets.json'), 'r'))
    try:
        with open(os.path.join(data_dir, 'region_descriptions.txt'), 'r') as f:
            region_descr = [l.strip() for l in f.readlines()]
    except FileNotFoundError:
        region_descr = json.load(open(os.path.join(data_dir, 'relationship_synsets.json'), 'r'))
        with open(os.path.join(data_dir, 'region_descriptions.txt'), 'w') as f:
            f.write('\n'.join([r['phrase'] for rd in region_descr for r in rd['regions']]))
    print('\n'.join(region_descr[:10]))
    print()

    preds = hd.predicates
    pred_verbs = [preds[0]] + [p.split('_')[0] for p in preds[1:]]
    predset = set(pred_verbs)
    objs_per_pred = {p: set() for p in preds}
    for i_r, r in enumerate(region_descr):
        if i_r % 1000 == 0:
            print(i_r)

        tokens = word_tokenize(r)
        tagged_tokens = pos_tag(tokens, tagset='universal')

        verb = None
        person_found = False
        for i_tokens, w in enumerate(tokens):
            if wn.morphy(w, wn.NOUN) in person_words:
                person_found = True
            else:
                verb = wn.morphy(w, wn.VERB)
                if verb in predset:
                    break
        else:
            continue

        assert verb is not None
        if not person_found or i_tokens + 1 == len(tokens):
            continue

        following_tagged_tokens = tagged_tokens[i_tokens + 1:]
        if following_tagged_tokens[0][1] == 'ADP':
            preposition = following_tagged_tokens[0][0]
            following_tagged_tokens = following_tagged_tokens[1:]
        else:
            preposition = None

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

        for i_pred, orig_p in enumerate(preds):
            if pred_verbs[i_pred] == verb:
                p_tokens = orig_p.split('_')
                if len(p_tokens) > 1 and (preposition is None or preposition != p_tokens[1]):
                    continue
                objs_per_pred[orig_p].add(p_obj)

    for p, objs in objs_per_pred.items():
        print('%20s:' % p, sorted(objs))

    with open(os.path.join(cfg.program.cache_root, 'vg_predicate_objects.pkl'), 'wb') as f:
        pickle.dump(objs_per_pred, f)

    print(len(region_descr))


if __name__ == '__main__':
    main()
