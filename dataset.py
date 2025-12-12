import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GalaxyDataset(Dataset):

    def __init__(self, h5_path, transform=None):
        self.transform = transform

        with h5py.File(h5_path, 'r') as f:
            images = f['images'][:]  ## (17736,256,256,3)
            ans = f['ans'][:]        ## (17736,)

        mask = np.isin(ans, [5,6,7])
        self.images = images[mask].astype(np.uint8)
        ans = ans[mask]

        self.labels = np.zeros(len(ans), dtype=np.int64)
        self.labels[ans == 5] = 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  ## (256,256,3)

        img = torch.tensor(img, dtype=torch.float).permute(2,0,1)
        img = img / 255.0 

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long) 

        return img, label

    def set_transform(self, transform):
        self.transform = transform


