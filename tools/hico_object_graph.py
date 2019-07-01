import os
from typing import List

import numpy as np

from config import cfg
from lib.dataset.hicodet.pc_hicodet_split import PrecomputedHicoDetSplit, HicoDetSplitBuilder, Splits


def export_to_rotate_edge_list(output_dir, edges, nodes: List[str], relations: List[str]):
    output = []
    for h, r, t in edges:
        output.append(f'{h}\t{r}\t{t}')
    lines = '\n'.join(output)
    with open(os.path.join(output_dir, 'train.txt'), 'w') as f:
        f.write(lines)
    with open(os.path.join(output_dir, 'entities.dict'), 'w') as f:
        f.write('\n'.join([f'{i}\t{n}' for i, n in enumerate(nodes)]))
    with open(os.path.join(output_dir, 'relations.dict'), 'w') as f:
        f.write('\n'.join([f'{i}\t{r}' for i, r in enumerate(relations)]))


def main():
    cfg.parse_args(fail_if_missing=False)
    cfg.data.val_ratio = 0

    # FIXME delete, it's for RotatE
    # self.entity_embedding = nn.Parameter(torch.from_numpy(np.load(os.path.join(emb_path, 'entity_embedding.npy'))))

    # train_split = HicoDetSplitBuilder.get_split(PrecomputedHicoDetSplit, split=Splits.TRAIN)  # type: PrecomputedHicoDetSplit
    test_split = HicoDetSplitBuilder.get_split(PrecomputedHicoDetSplit, split=Splits.TEST)  # type: PrecomputedHicoDetSplit
    train_split = test_split

    box_classes = np.concatenate([train_split.pc_box_labels,
                                  np.argmax(test_split.pc_boxes_ext[:, 5:], axis=1)
                                  ])
    box_im_ids = np.concatenate([train_split.pc_box_im_idxs,
                                 test_split.pc_box_im_idxs,
                                 ])
    im_ids, num_boxes_per_img = np.unique(box_im_ids, return_counts=True)
    assert np.all(im_ids == np.arange(im_ids.size))
    cum_num_boxes_per_img = np.cumsum(num_boxes_per_img)
    assert cum_num_boxes_per_img[-1] == box_im_ids.shape[0]
    cum_num_boxes_per_img[1:] = cum_num_boxes_per_img[:-1]
    cum_num_boxes_per_img[0] = 0
    ho_infos = cum_num_boxes_per_img[train_split.pc_ho_im_idxs, :] + train_split.pc_ho_infos[:, 1:]

    persons_per_interaction = {iid: [] for iid in range(train_split.hicodet.num_interactions)}
    for int_idx in range(train_split.pc_ho_infos.shape[0]):
        obj_class = train_split.pc_box_labels[train_split.pc_ho_infos[int_idx, 2]]
        actions = np.flatnonzero(train_split.pc_action_labels[int_idx, :])


    entities = [f'im{imid}_node{bid}' for bid, imid in enumerate(box_im_ids)]
    relations = [f'OnSameObj_{o}' for o in train_split.hicodet.objects] + \
                [f'Action_{a}' for a in train_split.hicodet.predicates]
    triples = []
    for i, pi in enumerate(box_im_ids):
        for j, pj in enumerate(hd.predicates):
            if i == j:
                continue

            common_objects = np.flatnonzero(p2p_from_op[i, j])
            triples += [[pi, f'Common_{hd.objects[o]}', pj] for o in common_objects]

            if syn_sim_mat[i, j] > 0:
                triples.append([pi, 'Synonym', pj])

            if sim_mat[i, j] > 0:
                triples.append([pi, 'Related', pj])

    # Check
    for p1, r, p2 in triples:
        assert p1 in entities and p2 in entities and r in relations

    export_to_rotate_edge_list('.', triples, entities, relations)


if __name__ == '__main__':
    main()
    # plot()
