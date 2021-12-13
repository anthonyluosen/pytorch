#-*-coding =utf-8 -*-
#@time :2021/12/5 10:48
#@Author: Anthony
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data.dataset import random_split
from math import ceil

def mydata():
    df_train = pd.read_csv('./data/new_shiny_train.csv')
    df_test = pd.read_csv('./data/new_shiny_test.csv')

    y = df_train['target'].values
    X = df_train.drop(['target', 'ID_code'],axis=1).values
    print ( X.shape)
    x_tensor = torch.tensor(X , dtype=torch.float32)
    y_tensor = torch.tensor(y , dtype= torch.float32)
    train_ds = TensorDataset(x_tensor,y_tensor)
    train_ids , valid_ids = random_split(train_ds, [int(len(train_ds)*0.9), ceil(len(train_ds)*0.1) ])

    X_test = df_test.drop(['ID_code'], axis=1).values
    x_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    y_testids = df_test['ID_code']
    test_ids = TensorDataset(x_tensor, y_tensor)

    return train_ids, valid_ids, test_ids, y_testids

if __name__ == '__main__':
    a , b , c ,d = mydata()
    print(len(a))
    print(len(b))
    print(len(c))
    print(d[1])
