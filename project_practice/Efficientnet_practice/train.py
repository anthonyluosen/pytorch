#-*-coding =utf-8 -*-
#@time :2021/12/6 11:45
#@Author: Anthony
import os
import torch
import torch.nn.functional as F
import numpy as np
from pytorch.Efficientnet_practice import config
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch.Efficientnet_practice.dataset import catdog
from efficientnet_pytorch import EfficientNet
from pytorch.Efficientnet_practice.utils import check_accuracy, load_checkpoint, save_checkpoint

def save_feature_vectors(model, loader, outputsize = (1,1), file= 'trainb0'):
    model.eval()
    images, labels= [], []
    for idx, (x,y) in enumerate(tqdm(loader)):
        x = x.to(config.Device)
        with torch.no_grad():
            features = model.extract_features(x)
            features = F.adaptive_avg_pool2d(features, output_size=outputsize)

        images.append(features.reshape(x.shape[0],-1).detach().cpu().numpy())
        labels.append(y.numpy())
    if not os.path.exists('./data/data_features'):
        os.makedirs('./data/data_features')
    np.save(f'./data/data_features/X_{file}.npy',np.concatenate(images,axis=0))
    np.save(f'./data/data_features/y_{file}.npy',np.concatenate(labels,axis=0))
    model.train()

def train_one_epoch(loader, model, loss_fn, optimizer, scaler):
    loop = tqdm(loader)
    for idx, (x,y ) in enumerate(loader):
        data = data.to(config.Device)
        targets = targets.to(config.Device).unsqueeze(1).float()

        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scaler.update()
            loop.set_postfix(loss = loss.item())

def main():
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(2560, 1)
    train_dataset = catdog(root="data/train/", transform=config.basic_transform)
    test_dataset = catdog(root="data/test/", transform=config.basic_transform)

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=config.BATCH_SIZE,
        num_workers=1,

    )
    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=config.BATCH_SIZE,
        num_workers=1
    )
    model = model.to(config.Device)
    # scaler = torch.cuda.amp.GradScaler()
    # loss = nn.BCEWithLogitsLoss()
    # optimizer = optim.Adam(
    #     model.parameters(), config.LEARNING_RATE,weight_decay = config.WEIGHT_DECAY
    # )
    # if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
    #     load_checkpoint(torch.load(config.CHECKPOINT_FILE),model)
    # for epoch in range(config.NUM_EPOCHS):
    #     train_one_epoch(train_loader, model, loss_fn=loss, optimizer=optimizer, scaler=scaler)
    #     check_accuracy(train_loader, model, loss)
    # if config.SAVE_MODEL:
    #     check_point = {'state_dict':model.state_dict(), 'optimizer':optimizer.state_dict()}
    #     save_checkpoint(check_point, filename=config.CHECKPOINT_FILE)

    save_feature_vectors(model, train_loader, outputsize=(1,1), file = 'train_b0')
    save_feature_vectors(model, test_loader, outputsize=(1,1), file = 'test_b0')


if __name__ == '__main__':
    main()


