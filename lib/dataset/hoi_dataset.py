import numpy as np


class HoiDataset:
    def __init__(self, object_classes, action_classes, null_action, interactions_classes):
        self.objects = object_classes
        self.object_index = {obj: i for i, obj in enumerate(self.objects)}

        # Actions
        self.null_action = null_action
        self.actions = action_classes
        self.action_index = {act: i for i, act in enumerate(self.actions)}
        assert self.action_index[self.null_action] == 0

        # Interactions
        self.interactions = np.array([[self.action_index[act], self.object_index[obj]] for act, obj in interactions_classes])  # [a, o]
        self.oa_pair_to_interaction = np.full([self.num_objects, self.num_actions], fill_value=-1, dtype=np.int)
        self.oa_pair_to_interaction[self.interactions[:, 1], self.interactions[:, 0]] = np.arange(self.num_interactions)

    @property
    def num_objects(self):
        return len(self.objects)

    @property
    def num_actions(self):
        return len(self.actions)

    @property
    def num_interactions(self):
        return self.interactions.shape[0]

    @property
    def interaction_to_object_mat(self):
        interactions_to_obj = np.zeros((self.num_interactions, self.num_objects))
        interactions_to_obj[np.arange(self.num_interactions), self.interactions[:, 1]] = 1
        return interactions_to_obj

    @property
    def interaction_to_action_mat(self):
        interactions_to_act = np.zeros((self.num_interactions, self.num_actions))
        interactions_to_act[np.arange(self.num_interactions), self.interactions[:, 0]] = 1
        return interactions_to_act

    def get_img_dir(self, split):
        raise NotImplementedError
