import torch

class PredictionDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data = self.dataset[index]

        if isinstance(data, (tuple, list)):
            return data[0]
        
        return data

    def __len__(self):
        return len(self.dataset)