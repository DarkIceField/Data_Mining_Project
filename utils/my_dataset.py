from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import List
from torchvision import transforms
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, csv_path: str, img_transform=transforms.ToTensor()):
        super(MyDataSet, self).__init__()
        self.tabular = pd.read_csv(csv_path)
        self.images_path = self.tabular["path"].values
        scaler = StandardScaler()
        self.features = scaler.fit_transform(
            self.tabular.iloc[:, 1:-13].values.astype(np.float32)
        )
        self.labels = self.tabular.iloc[:, -13:-7].values.astype(np.float32)
        self.img_transform = img_transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img_path = self.images_path[item]
        img = Image.open(img_path)
        # RGB为彩色图片，L为灰度图片
        if img.mode != "RGB":
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        img = self.img_transform(img)
        aux = torch.from_numpy(self.features[item])
        label = torch.from_numpy(self.labels[item])

        return img, aux, label

class MyEvalDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, csv_path: str, img_transform=transforms.ToTensor()):
        super(MyEvalDataSet, self).__init__()
        self.tabular = pd.read_csv(csv_path)
        self.images_path = self.tabular["path"].values
        scaler = StandardScaler()
        self.features = scaler.fit_transform(
            self.tabular.iloc[:, 1:-1].values.astype(np.float32)
        )
        self.img_transform = img_transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img_path = self.images_path[item]
        img = Image.open(img_path)
        # RGB为彩色图片，L为灰度图片
        if img.mode != "RGB":
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        img = self.img_transform(img)
        aux = torch.from_numpy(self.features[item])
        _,img_name = os.path.split(img_path)
        id,_ = os.path.splitext(img_name)

        return img, aux,id

if __name__ == "__main__":
    # unit test
    data_root = os.path.abspath(os.path.join(os.getcwd(), "data", "train"))
    dataset = MyDataSet(
        csv_path=os.path.join(data_root, "train.csv"),
    )
    for index, (img, aux, label) in enumerate(dataset):
        if index < 10:
            print(img.shape)
            print(aux.shape, aux.dtype)
            # print(aux)
            print(label.shape, label.dtype)
            # print(label)
