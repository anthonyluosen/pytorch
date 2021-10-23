#-*-coding =utf-8 -*-
#@time :2021/10/22 21:23
#@Author: Anthony
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import Compose
import torch
from  torch.utils.tensorboard  import summary
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch.WGANs.model import Generator,Discrimnator,initialize_weight

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 2e-4
channel_img = 1
batch_size = 64
features_g = 64
features_d = 64
noise_dim = 100
img_size = 64
epochs = 5

