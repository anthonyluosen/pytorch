#-*-coding =utf-8 -*-
#@time :2021/10/23 11:59
#@Author: Anthony
import torch
import torchvision
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import Compose
import pandas as pd
import math

class Mywine(Dataset):
    def __init__(self,transform = None):
        self.transform = transform
        data = np.loadtxt('wine.csv',skiprows=1,delimiter=',',dtype=np.float32)
        self.x = data[:,1:]
        self.y = data[:,[0]]
        self.n_samples = data.shape[0]

    def __getitem__(self, item):
        sample = self.x[item] , self.y[item]
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.n_samples

class ToTensor:
    def __call__(self, sample):
        inputs , target = sample
        return torch.from_numpy(inputs),torch.from_numpy(target)
class MultiTransform:
    def __init__(self,factor):
        self.factor = factor
    def __call__(self, sample):
        input , target = sample
        input *= self.factor
        return input,target
data = Mywine()
first_data = data[0]
inputs , features = first_data
print(type(inputs))
print(inputs)
composed = Compose(
   [ ToTensor(),
     MultiTransform(2)]
)
data1 = Mywine(transform=composed)
first_data1 = data1[0]
inputs1 , features1 = first_data1
print(type(inputs1))
print(inputs1)

