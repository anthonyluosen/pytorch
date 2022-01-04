#-*-coding =utf-8 -*-
#@time :2021/10/12 14:54
#@Author: Anthony
import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_Text(nn.Module):
    def __init__(self,vocab_size,embedding_dim,output_dim,kernel_size,dropout,filer_size,drop):
        super(CNN_Text, self).__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.cnn = nn.ModuleList([
            nn.Conv2d(
                in_channels=1,out_channels=filer_size,
                kernel_size=(kernel,embedding_dim)
            ) for kernel in kernel_size
        ])

        self.linear1 = nn.Linear(filer_size*len(kernel_size) , 128)
        self.linaer2 = nn.Linear(128,output_dim)
        self.dropout = nn.Dropout(drop)

    def forward(self,x):
        # [sentence len,batch_size]
        x = x.permute(1,0)
        x = self.embedding(x)
        #[batch size , sentence len,embedding_dim]
        x = x.unsqueeze(1)
        #[batch size ,1, sentence len,embedding_dim]
        conved = [F.relu(conv(x)).squeeze(3) for conv in self.cnn]

        x = self.dropout(torch.cat(conved,dim=1))
        #[ drop size , sentence len*len(kernel_size)]


