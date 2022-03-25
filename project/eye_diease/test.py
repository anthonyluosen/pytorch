import torch

import config
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
images_folder = 'D:\database\eye\\train'
image = np.array(Image.open(os.path.join(images_folder, '17_left'+".jpeg")))
plt.imshow(image)
plt.show()
print(image.shape)

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),

    # 随机缩放裁剪 size 224*224
    # torchvision.transforms.RandomResizedCrop(224),
    # 随机裁剪 size 224*224
    # torchvision.transforms.RandomCrop(224),
    # 中心裁剪 size 224*224
    # torchvision.transforms.CenterCrop(224),
    # 将图片的尺寸 Resize 到128*128 不裁剪
    torchvision.transforms.Resize((230, 230)),
    # 转为张量并归一化到[0,1]（是将数据除以255），且会把H*W*C会变成C *H *W
    # 数据归一化处理，3个通道中的数据整理理到[-1, 1]区间。3个通道，故有3个值。该[0.485, 0.456, 0.406]这一组平均值是从imagenet训练集中抽样算出来的。
    # ToTensor（）的[0，1]只是范围改变了， 并没有改变分布，mean和std处理后可以让数据正态分布
    torchvision.transforms.Normalize(mean=[0.3199, 0.2240, 0.1609],
                                     std=[0.3020, 0.2183, 0.1741],
                                     ),
    torchvision.transforms.ToTensor
])
print(data_transform(image))