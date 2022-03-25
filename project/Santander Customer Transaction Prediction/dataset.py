#-*-coding =utf-8 -*-
#@time :2021/12/5 10:48
#@Author: Anthony
import config
import pandas as pd
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from math import ceil
from tqdm import tqdm
# class DRDataset(Dataset):
#     def __init__(self, images_folder, path_to_csv, train=True, transform=None):
#         super().__init__()
#         self.data = pd.read_csv(path_to_csv)
#         self.images_folder = images_folder
#         self.image_files = os.listdir(images_folder)
#         self.transform = transform
#         self.train = train
#
#     def __len__(self):
#         return self.data.shape[0] if self.train else len(self.image_files)
#
#     def __getitem__(self, index):
#         if self.train:
#             image_file, label = self.data.iloc[index]
#         else:
#             # if test simply return -1 for label, I do this in order to
#             # re-use same dataset class for test set submission later on
#             image_file, label = self.image_files[index], -1
#             image_file = image_file.replace(".jpeg", "")
#
#         image = np.array(Image.open(os.path.join(self.images_folder, image_file+".jpeg")))
#
#         if self.transform:
#             image = self.transform(image=image)["image"]
#
#         return image, label, image_file

class DRDataset(Dataset):
    def __init__(self,path_to_csv,data_dir,train = True, transformer = None):
        super().__init__()
        self.data = pd.read_csv(path_to_csv)
        self.imager_folder = os.listdir(data_dir)
        self.image_dir = data_dir
        self.transform = transformer
        self.train  = train

    def __len__(self):
        return self.data.shape[0] if self.train else len(self.imager_folder)

    def __getitem__(self, item):
        if self.train:
            image_file,image_label = self.data.iloc[item]
        else:
            # if test set simple return -1 as the label
            image_file ,image_label = self.imager_folder[item],-1
            image_file = image_file.replace('.jpeg','')
        img = np.array(Image.open(os.path.join(self.image_dir,image_file+'.jpeg' )))
        if self.transform:
            img = self.transform(image = img)['image']
            # print('transform done!')
        return img, image_label, image_file


if __name__ == '__main__':
    path_train = 'D:\\database\\eye\\train'
    path_label = 'D:\\database\\eye\\trainLabels.csv'
    dataset = DRDataset(
        path_to_csv=path_label,
        data_dir=path_train,
        train = True,
        transformer=config.val_transforms
    )
    loader = DataLoader(dataset, batch_size=12, num_workers=2, shuffle=True,pin_memory=True)
    for x,label,file in tqdm(loader):
        print(x.shape)
        print(label)
        import sys
        sys.exit()

