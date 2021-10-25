#-*-coding =utf-8 -*-
#@time :2021/10/22 21:23
#@Author: Anthony
import torchvision
import torchvision.datasets as datasets
from torchvision.transforms import Compose
import torch
import torch.nn as nn
from  torch.utils.tensorboard  import SummaryWriter
from torchvision import transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch.WGANs.model import Generator,Discrimnator,initialize_weight
# writer  = SummaryWriter(f'log/')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
learning_rate = 5e-5
channel_img = 3
batch_size = 64
features_g = 64
features_d = 64
noise_dim = 128
img_size = 64
epochs = 5
disc_iterations = 5
weight_clip = 0.01

transform = Compose(
    [
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5 for _ in range(channel_img)] ,[0.5 for _ in range(channel_img)] )
    ]
)
dataset = datasets.ImageFolder(root='' , transform=transform,)
# data = datasets.MNIST(root='datasets' , download=True , transform=transform , train=True)
dataload = DataLoader(dataset=dataset ,batch_size=batch_size , shuffle=True)

disc = Discrimnator(chnnels_img=channel_img , features_d=features_d).to(device)
gen = Generator(noise_dim=noise_dim , feature_g=features_g , channels_img=channel_img).to(device)

initialize_weight(disc)
initialize_weight(gen)

opt_gen = optim.RMSprop(gen.parameters(),lr=learning_rate)
opt_disc = optim.RMSprop(disc.parameters(),lr=learning_rate)

fixed_noise = torch.randn((32,noise_dim,1,1)).to(device)
criterion = nn.BCELoss()

writer_real = SummaryWriter(f'log/real')
writer_fake = SummaryWriter(f'log/fake')
step = 0

disc.train()
gen.train()

for epoch in range(epochs):
    for idx, (real , _) in enumerate(dataload):
        real = real.to(device)
        # noise = torch.randn(batch_size , noise_dim,1,1).to(device)
        cur_batch_size = real.shape[0]
        # fake = gen(noise)

        #train the discriminator first max(log(D(x)+log(1-D(G(z))))
        for _ in range(disc_iterations):
            noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)
            fake = gen(noise)
            disc_real = disc(real).reshape(-1)
            disc_fake = disc(fake).reshape(-1)

            loss_d = -(torch.mean(disc_real)-torch.mean(disc_fake))

            disc.zero_grad()
            loss_d.backward(retain_graph = True)
            opt_disc.step()

            # clip critic weight between -0.01,0.01
            for p in disc.parameters():
                p.data.clamp_(-weight_clip,weight_clip)

        # Train Generator: max E[critic(gen_fake)] <-> min -E[critic(gen_fake)]
        output = disc(fake).reshape(-1)
        loss_g = -torch.mean(output)
        gen.zero_grad()
        loss_g.backward()
        opt_gen.step()

        # print losses occasionally and print it to tensorboard
        if idx %100 == 0 and idx > 0 :
            disc.eval()
            gen.eval()
            print(f"epoch [{epoch}/{epochs}], Batch {idx}/{len(dataload)} \
                  loss_D {loss_d:.4f} , loss_G {loss_g:.4f}"
                  )
            with torch.no_grad():
                fake = gen(fixed_noise)
                # take out 32 example
                img_real_grid  = torchvision.utils.make_grid(
                    real[:32] , normalize=True
                )
                img_fake_grid = torchvision.utils.make_grid(
                    fake[:32] , normalize=True
                )
                writer_real.add_image('real',img_real_grid,global_step=step)
                writer_fake.add_image('fake',img_fake_grid,global_step=step)
            step+=1
            gen.train()
            disc.train()








