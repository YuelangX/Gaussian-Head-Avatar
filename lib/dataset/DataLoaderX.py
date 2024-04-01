import torch
from prefetch_generator import BackgroundGenerator

class DataLoaderX(torch.utils.data.DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())