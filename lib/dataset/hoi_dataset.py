import torch
from torch.utils.data import Dataset


class HoiDatasetSplit(Dataset):
    @property
    def human_class(self) -> int:
        raise NotImplementedError

    @property
    def num_object_classes(self):
        raise NotImplementedError

    @property
    def num_predicates(self):
        raise NotImplementedError

    @property
    def num_interactions(self):
        raise NotImplementedError

    @property
    def num_images(self):
        raise NotImplementedError

    def get_loader(self, batch_size, num_workers=0, num_gpus=1, shuffle=None, drop_last=True, **kwargs) -> torch.utils.data.DataLoader:
        raise NotImplementedError
