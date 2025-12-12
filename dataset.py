import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class GalaxyDataset(Dataset):

    def __init__(self, h5_path, transform=None):
        self.transform = transform

        ## 一定要加载所有数据到内存！只打开一次
        with h5py.File(h5_path, 'r') as f:
            images = f['images'][:]  ## (17736,256,256,3)
            ans = f['ans'][:]        ## (17736,)

        ## 选择ans=567的源(barred， unbarred)
        mask = np.isin(ans, [5,6,7])
        self.images = images[mask].astype(np.uint8)
        ans = ans[mask]

        ## 定义二分类标签，barred的label是1
        self.labels = np.zeros(len(ans), dtype=np.int64)
        self.labels[ans == 5] = 1

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]  ## (256,256,3)

        img = torch.tensor(img, dtype=torch.float).permute(2,0,1)  ## (3,256,256)
        img = img / 255.0  ## 如果原图是 uint8，归一化到 [0,1]

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(self.labels[idx], dtype=torch.long) ##标签转 Tensor

        return img, label

    def set_transform(self, transform):
        self.transform = transform

