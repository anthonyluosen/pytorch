#-*-coding =utf-8 -*-
#@time :2021/10/23 9:59
#@Author: Anthony
import torch
import torchvision
import numpy as np
import math
import pandas as pd
from torch.utils.data import DataLoader ,Dataset
# path = 'https://raw.githubusercontent.com/python-engineer/pytorchTutorial/master/data/wine/wine.csv'
# data = pd.read_csv(path)
# data.to_csv('wine.csv')
# print(data.head())
class WineDataset(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # here the first column is the class label, the rest are the features
        self.x_data = torch.from_numpy(xy[:, 1:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [0]]) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples


# create dataset
dataset = WineDataset()

# get first sample and unpack
first_data = dataset[0]
features, labels = first_data
print(features, labels)

# Load whole dataset with DataLoader
# shuffle: shuffle data, good for training
# num_workers: faster loading with multiple subprocesses
# !!! IF YOU GET AN ERROR DURING LOADING, SET num_workers TO 0 !!!
train_loader = DataLoader(dataset=dataset,
                          batch_size=4,
                          shuffle=True,
                          num_workers=0)

# convert to an iterator and look at one random sample
# dataiter = iter(train_loader)
# data = dataiter.__next__()
# features, labels = data
# print(features, labels)

epochs = 2
iteration = math.ceil(len(dataset)/4)
for epoch in range(epochs):
    for idx ,(input,label) in enumerate(train_loader):
        print(f'epoch {epoch+1}/{epochs}  ,step {idx+1}/{iteration},inputs {input.shape}')

