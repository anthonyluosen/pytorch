#-*-coding =utf-8 -*-
#@time :2021/12/5 10:47
#@Author: Anthony
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
from pytorch.project_practice.dataset import  mydata
from pytorch.project_practice.utils import  get_predictions, get_submission
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F
from  torch.optim import Adam

class NN(nn.Module):
    def __init__(self, inputsize, hidden_size):
        super(NN, self).__init__()
        self.bn = nn.BatchNorm1d(inputsize*2)
        self.fn1 = nn.Linear(2,hidden_size)
        self.fn2 = nn.Linear(inputsize*hidden_size, 1)

        # self.net = nn.Sequential(
        #     nn.BatchNorm1d(inputsize),
        #     nn.Linear(inputsize,50),
        #     nn.ReLU(),
        #     nn.Linear(50,1)
        # )
    def forward(self,x):
        batch_size = x.shape[0]
        out = self.bn(x)
        feature1 = out[:, :200].unsqueeze(2)
        feature2 = out[:, 200:].unsqueeze(2)
        feature = torch.cat([feature1, feature2] ,dim=2)
        out = F.relu(self.fn1(feature))
        out = out.view(batch_size,-1)
        # shape was [batch, 200, hidden_size]
        # here we get [batch, input_size*hidden_size]
        # [batch_size , 200]
        # out = F.relu(self.fn1(out)).reshape(batch_size, -1)
        #[batch_size , input_size * hidden_size]
        return torch.sigmoid(self.fn2(out).view(-1))

train_ds, valid_ids, test_ids, test_id = mydata()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = NN(200,64).to(device)

optimzier = Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
lossfn = nn.BCELoss()

train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
vlaid_loader = DataLoader(valid_ids, batch_size=128)
test_loader = DataLoader(test_ids, batch_size=64)

x,y = next(iter(train_loader))

for epoch in range(20):
    probablities , true_label = get_predictions(loader=vlaid_loader, model=model,
                                                device=device)
    print(f'valid roc {roc_auc_score(true_label,probablities)}')
    # x, y = next(iter(train_loader))
    for batch_idx, (x,y) in enumerate(train_loader):
        x.to(device)
        y.to(device)

        scores = model(x)
        loss = lossfn(scores, y)
        optimzier.zero_grad()
        loss.backward()
        optimzier.step()

get_submission(test_loader, model, device, test_id)

