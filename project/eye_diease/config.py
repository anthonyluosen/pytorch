#-*-coding =utf-8 -*-
#@time :2021/12/17 14:26
#@Author: Anthony
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 3e-5
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 20
NUM_EPOCHS = 100
NUM_WORKERS = 6
CHECKPOINT_FILE = "b3.pth.tar"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True

# Data augmentation for images
train_transforms = A.Compose(
    [
        A.Resize(width=230, height=230),
        A.RandomCrop(height=728, width=728),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Blur(p=0.3),
        A.CLAHE(p=0.3),
        A.ColorJitter(p=0.3),
        A.CoarseDropout(max_holes=12, max_height=20, max_width=20, p=0.3),
        A.IAAAffine(shear=30, rotate=0, p=0.2, mode="constant"),
        A.Normalize(
            mean=[0.3199, 0.2240, 0.1609],
            std=[0.3020, 0.2183, 0.1741],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

val_transforms = A.Compose(
    [
        A.Resize(height=230, width=230),
        A.Normalize(
            mean=[0.3199, 0.2240, 0.1609],
            std=[0.3020, 0.2183, 0.1741],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

import torchvision

data_transform = torchvision.transforms.Compose([
    # 随机缩放裁剪 size 224*224
    # torchvision.transforms.RandomResizedCrop(224),
    # 随机裁剪 size 224*224
    # torchvision.transforms.RandomCrop(224),
    # 中心裁剪 size 224*224
    # torchvision.transforms.CenterCrop(224),
    # 将图片的尺寸 Resize 到128*128 不裁剪
    torchvision.transforms.Resize((230, 230)),
    # 转为张量并归一化到[0,1]（是将数据除以255），且会把H*W*C会变成C *H *W
    torchvision.transforms.ToTensor(),
    # 数据归一化处理，3个通道中的数据整理理到[-1, 1]区间。3个通道，故有3个值。该[0.485, 0.456, 0.406]这一组平均值是从imagenet训练集中抽样算出来的。
    # ToTensor（）的[0，1]只是范围改变了， 并没有改变分布，mean和std处理后可以让数据正态分布
    torchvision.transforms.Normalize(mean=[0.3199, 0.2240, 0.1609],
                                     std=[0.3020, 0.2183, 0.1741],
                                     ),
])