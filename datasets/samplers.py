import torch
import random
import math, random
from collections import defaultdict
from torch.utils.data import Sampler, Subset

class ScaleBatchSampler(Sampler):
    def __init__(self, dataset_size, batch_size, scale_min, scale_max):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.scale_min = scale_min
        self.scale_max = scale_max

    def __iter__(self):
        indices = torch.randperm(self.dataset_size).tolist()

        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i+self.batch_size]

            s = random.uniform(self.scale_min, self.scale_max)

            yield [(idx, s) for idx in batch_indices]

    def __len__(self):
        return self.dataset_size // self.batch_size

class SameSizeBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=False):
        if isinstance(dataset, Subset):
            self.subset_indices = list(dataset.indices)  # índices no dataset original
            self.base_dataset   = dataset.dataset
        else:
            self.subset_indices = list(range(len(dataset)))
            self.base_dataset   = dataset

        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.H = self.base_dataset.hr_data.shape[1]
        self.W = self.base_dataset.hr_data.shape[2]
        self.scale_min = self.base_dataset.scale_min
        self.scale_max = self.base_dataset.scale_max

    def _build_groups(self):
        groups = defaultdict(list)  # (h_lr, w_lr) -> [(pos_no_subset, orig_idx, scale)]
        for pos, orig_idx in enumerate(self.subset_indices):
            s     = random.uniform(self.scale_min, self.scale_max)
            h_lr  = math.floor(self.H / s + 1e-9)
            w_lr  = math.floor(self.W / s + 1e-9)
            groups[(h_lr, w_lr)].append((pos, orig_idx, s))
        return groups

    def __iter__(self):
        groups = self._build_groups()

        # assign_scales usa orig_idx porque é o que SRTinyDataset.__getitem__ recebe
        scale_map = {orig_idx: s for items in groups.values() for _, orig_idx, s in items}
        self.base_dataset.assign_scales(scale_map)

        batches = []
        for items in groups.values():
            positions = [pos for pos, _, _ in items]  # pos no Subset (0..len(subset)-1)
            random.shuffle(positions)
            for i in range(0, len(positions), self.batch_size):
                batch = positions[i : i + self.batch_size]
                if self.drop_last and len(batch) < self.batch_size:
                    continue
                batches.append(batch)

        random.shuffle(batches)
        yield from batches

    def __len__(self):
        n = len(self.subset_indices)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)