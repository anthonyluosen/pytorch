#-*-coding =utf-8 -*-
#@time :2021/12/5 10:47
#@Author: Anthony
import torch
import pandas as pd
import numpy as np
import tqdm

import config
import warnings
import torch.nn.functional as F

def make_predictions(model, laoder, output = 'submission.csv'):
    preds = []
    filenames = []
    model.eval()

    for x, y,files in tqdm.tqdm(laoder):
        x = x.to(config.DEVICE)
        with torch.no_grad():
            pred = model(x).argmax(1)
            preds.append(pred.cpu().numpy())
            filenames += files
    df = pd.DataFrame({'image':filenames, 'level':np.concatenate(preds,axis=0)})
    df.to_csv(output, index=False)
    model.train()
def check_accuracy(loader, model, device =' cuda'):
    model.eval()
    all_preds, all_labels = [], []
    num_correct = 0
    num_samples = 0

    for x,y,file in tqdm.tqdm(loader):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        with torch.no_grad():
            _,predictions = model(x)
            num_correct += (predictions==y).sum()
            num_samples += predictions.shape[0]
            all_preds.append(predictions.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    print(f'Got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    model.train()
    return np.concatenate(all_preds, axis = 0, dtype=np.int64),np.concatenate(
        all_labels,axis=0,dtype=np.int64
    )
def save_checkpoint(state, filename = 'my_checkpoint.tar'):
     print('>>> save the check point !')
     torch.save(state,filename)

def load_checkpoint(check_point, model, optimizer, lr):
    print('>>>load the check point!')
    model.load_state_dict(check_point['state_dict'])
    optimizer.load_state_dict(check_point['optimizer'])

    # if we don't do this then it just have the learning rate of old checkpoint
    # and it will lead to many hours of debugging
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

