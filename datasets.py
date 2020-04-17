from torch.utils.data import IterableDataset, Dataset
import numpy as np


class TSPTrainSet(IterableDataset):
    """
    Generate TSP instances on the fly
    """
    def __init__(self, n_batch_per_epoch, batch_size, n_node):
        super().__init__()
        self.n_node = n_node
        self.batch_size = batch_size
        self.n_batch_per_epoch = n_batch_per_epoch

    def __generator(self):
        for _ in range(self.n_batch_per_epoch):
            yield np.random.uniform(size=(self.batch_size, self.n_node,
                                          2)).astype(np.float32)

    def __iter__(self):
        return iter(self.__generator())

    def __len__(self):
        return self.n_batch_per_epoch


class TSPTestSet(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
