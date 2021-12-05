#-*-coding =utf-8 -*-
#@time :2021/12/5 11:50
#@Author: Anthony
import pandas as pd
df = pd.read_csv('new_train.csv')

print(df.head())

# df2 = pd.read_csv('new_test.csv')
# df2.drop('Unnamed: 0',axis=1,inplace=True)
# df2.to_csv('new_test.csv')