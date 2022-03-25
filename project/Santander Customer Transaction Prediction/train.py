#-*-coding =utf-8 -*-
#@time :2021/12/5 10:47
#@Author: Anthony
import torch
import torch.nn as nn
import os

from tqdm import tqdm
from torch.utils.data import DataLoader
import config
from sklearn.metrics import cohen_kappa_score
from efficientnet_pytorch import EfficientNet
from torchvision.utils import  save_image
import  torch.optim  as optim
from dataset import DRDataset
from utils import (make_predictions, check_accuracy, load_checkpoint,save_checkpoint
                   )


def train_one_epoch(loader, model, optimizer, loss_fn, scaler, device):
    losses = []
    loop = tqdm(loader)
    for batch_idx, (data, targets, _) in enumerate(loop):
        # save examples and make sure they look ok with the data augmentation,
        # tip is to first set mean=[0,0,0], std=[1,1,1] so they look "normal"
        #save_image(data, f"hi_{batch_idx}.png")

        data = data.to(device)
        targets = targets.to(device=device)

        # forward
        with torch.cuda.amp.autocast():
            scores = model(data)
            loss = loss_fn(scores, targets)

        losses.append(loss.item())

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loop.set_postfix(loss=loss.item())

    print(f"Loss average over epoch: {sum(losses)/len(losses)}")
def main():
    train_ds = DRDataset(
        data_dir='D:\\database\\eye\\train',
        path_to_csv='D:\\database\\eye\\train.csv',
        transformer=config.bsae_transforms
    )
    val_ds = DRDataset(
        data_dir='D:\\database\\eye\\train',
        path_to_csv='D:\\database\\eye\\vallabels.csv',
        transformer=config.bsae_transforms
    )
    test_ids = DRDataset(
        data_dir='D:\\database\\eye\\test',
        path_to_csv='D:\\database\\eye\\trainLabels.csv',
        transformer=config.val_transforms,
        train=False
    )
    train_loader = DataLoader(train_ds,
                              batch_size=config.BATCH_SIZE,
                              num_workers=0,
                              pin_memory=False,
                              shuffle=True
                              )
    test_loader = DataLoader(test_ids,
                              batch_size=config.BATCH_SIZE,
                              num_workers=0,
                              pin_memory=False,
                              shuffle=True
                              )
    test_loader = DataLoader(val_ds,
                             batch_size=config.BATCH_SIZE,
                             num_workers=0,
                             pin_memory=False,
                             shuffle=False
                             )

    loss_fn = nn.CrossEntropyLoss()
    model = EfficientNet.from_pretrained('efficientnet-b0')
    model._fc = nn.Linear(1280,5)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scaler = torch.cuda.amp.GradScaler()

    model.to(config.DEVICE)
    if config.LOAD_MODEL and config.CHECKPOINT_FILE in os.listdir():
        load_checkpoint(torch.load(config.CHECKPOINT_FILE),model,optimizer,config.LEARNING_RATE)

    for epoch in range(config.NUM_EPOCHS):
        train_one_epoch(train_loader, model, optimizer, loss_fn, scaler, config.DEVICE)

        #get one validation
        preds,labels = check_accuracy(val_ds,model,config.DEVICE)
        print(f'QuadraticWeightedKappa(validation):{cohen_kappa_score(labels, preds,weights="quadratic")}')

        if config.SAVE_MODEL:
            check_point = {
                'state_dict':model.state_dict(),
                'optimizer':optimizer.state_dict(),

            }
            save_checkpoint(check_point,filename=config.CHECKPOINT_FILE)
    make_predictions(model,test_loader)

if __name__ == '__main__':
    # print(config.DEVICE)
    main()