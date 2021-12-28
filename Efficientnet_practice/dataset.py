#-*-coding =utf-8 -*-
#@time :2021/12/6 11:16
#@Author: Anthony
import os
import re
import numpy as np
from torch.utils.data import Dataset
from PIL import Image

class catdog(Dataset):
    def __init__(self,root, transform):
        self.images = os.listdir(root)
        # print(self.images)
        self.images.sort(key=lambda x:int(re.findall(r'\d+',x)[0]))
        # print(self.images)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        file = self.images[item]
        img = np.array(Image.open(os.path.join(self.root, file)))

        if self.transform is not None:
            img = self.transform(image = img)['image']
        if 'dog' in file:
            label = 1
        elif 'cat' in file:
            label = 0
        else:
            label = -1

        return img, label

if __name__  == '__main__':
    dataset = catdog('./data/train/', transform=None)
    a, b = dataset.__getitem__(1)
    print(a.shape, b)

