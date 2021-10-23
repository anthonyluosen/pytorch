#-*-coding =utf-8 -*-
#@time :2021/10/22 19:43
#@Author: Anthony
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self,noise_dim , feature_g,channels_img):
        super().__init__()
        self.gen = nn.Sequential(
            self._block(noise_dim , feature_g*16,4,2,0),           #4*4
            self._block(feature_g*16, feature_g*8 , 4, 2, 1),    #8x8
            self._block(feature_g*8, feature_g* 4, 4, 2, 1),         #16x16
            self._block(feature_g*4, feature_g*2, 4, 2, 1),          #32x32
            nn.ConvTranspose2d(feature_g*2,channels_img , kernel_size=4,stride=2,padding=1),
            #64x64
            nn.Tanh()
        )

    def _block(self,in_channels , out_channels , kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels,
                      out_channels,
                      kernel_size,
                      stride,
                      padding,
                      bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()

        )
    def forward(self,x):
        return self.gen(x)

class Discrimnator(nn.Module):
    def __init__(self,chnnels_img, features_d, ):
        super(Discrimnator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(chnnels_img,features_d , kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d,features_d*2,4,2,1),
            self._block(features_d*2,features_d*4,4,2,1),
            self._block(features_d*4,features_d*8,4,2,1),
            nn.Conv2d(features_d*8,1,4,2,0),


        )
    def _block(self,in_channels,out_channels, kernel_size, stride,padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2))

    def forward(self,x):
        return self.disc(x)
def initialize_weight(model):
    for p in model.modules:
        if isinstance(p,(nn.Conv2d , nn.ConvTranspose2d,nn.BatchNorm2d)):
            nn.init.normal_(p.weight.data,0.0,0.02)
def test():
    batch, img_channel, w, d = 8,1,64,64
    img = torch.randn((batch,img_channel,w,d))
    print(img.shape)
    dics  = Discrimnator(img_channel,features_d=8)
    print(dics(img).shape)
    assert dics(img).shape == (batch,1,1,1) ,'discrimnator was failed'
    noise = torch.randn((batch,100,1,1))
    print(noise.shape)
    noise_dim = 100
    gen = Generator(noise_dim=noise_dim,feature_g=64, channels_img = img_channel)
    print(gen(noise).shape)
    assert gen(noise).shape  == (batch,img_channel,w, d) ,'generator was failed'
if __name__ == '__main__':
    test()