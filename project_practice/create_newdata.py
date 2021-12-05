#-*-coding =utf-8 -*-
#@time :2021/12/5 11:02
#@Author: Anthony
import pandas as pd
from math import ceil
df_train = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')

train_sample = df_train.sample(frac=0.3)
test_sample = df_test.sample(frac=0.3)
train_sample.to_csv('./data/new_train.csv')
test_sample.to_csv('./data/new_test.csv')