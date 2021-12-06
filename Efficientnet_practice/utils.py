#-*-coding =utf-8 -*-
#@time :2021/12/6 15:59
#@Author: Anthony
import torch
import os
import pandas as pd
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pytorch.Efficientnet_practice.dataset import catdog
import pytorch.Efficientnet_practice.config as config
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import log_loss

def check_accuracy(
        loader,model,loss_fn,input_shape = None,
        toggle_eval = True, print_accuracy = True
):
    '''
    :return: accuracy from the model:validation set
    '''
    if toggle_eval:
        model.eval()
    device = next(model.parameters()).device
    num_correct = 0
    num_samples = 0

    y_preds = []
    y_true = []
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            if input_shape:
                x = x.reshape(x.shape[0],*input_shape)
            scores = model(x)
            predictions = torch.sigmoid(scores)>0.5
            y_preds.append(torch.clip(torch.sigmoid(scores),0.005,0.995).cpu().numpy())
            y_true.append(y.cpu().numpy())
            num_correct += (predictions.squeeze(1)==y).sum()
            num_samples += predictions.size(0)

        accuracy = num_correct/num_samples
        if toggle_eval:
            model.train()

        if print_accuracy:
            print(f"Accuracy: {accuracy * 100:.2f}%")
            print(log_loss(np.concatenate(y_true, axis=0), np.concatenate(y_preds, axis=0)))
        return accuracy

def save_checkpoint(state, filename = 'my_checkpoint.pth.tar'):
    print('=> saving checkpoint')
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print('=> load checkpoint')
    model.load_state_dict(checkpoint['state_dict'])

def create_submission(model,model_name, files_dir):
    my_transforms = {
        'base':A.Compose(
            [
                A.Resize(height=240, width=240),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        ),
        'horizontal_flip':A.Compose(
            [
                A.Resize(height=240, width=240),
                A.HorizontalFlip(p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        ),
        'vectical_flip':A.Compose(
            [
                A.Resize(height=240, width=240),
                A.VerticalFlip(p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        ),
        'coloring': A.Compose(
            [
                A.Resize(height=240, width=240),
                A.ColorJitter(p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        ),
        'rotate': A.Compose(
            [
                A.Resize(height=240, width=240),
                A.Rotate(p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        ),
        'shear': A.Compose(
            [
                A.Resize(height=240, width=240),
                A.IAAAffine(p=1.0),
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ]
        ),
    }

    for t in ["base", "horizontal_flip", "vertical_flip", "coloring", "rotate", "shear"]:
        predictions = []
        labels = []
        all_files = []
        test_data = catdog(root=files_dir, transform=my_transforms[t])
        test_loader = DataLoader(
            test_data, batch_size=32, num_workers=1, shuffle=False,
        )
        model.eval()
        for idx, (x,y,file_names) in enumerate(test_loader):
            x = x.to(config.Device)
            with torch.no_grad():
                outputs = torch.clip(torch.sigmoid(model(x)), 0.005,0.995).squeeze(1).cpu().numpy()
                predictions.append(outputs)
                labels += file_names
        df = pd.DataFrame(
            {
                "id": np.arange(
                    1,
                    (len(predictions) - 1) * predictions[0].shape[0]
                    + predictions[-1].shape[0]
                    + 1,
                ),
                "label": np.concatenate(predictions, axis=0),
            }
        )
        df.to_csv(f"predictions_test/submission_{model_name}_{t}.csv", index=False)

        model.train()
        print(f"Created submission file for model {model_name} and transform {t}")
def blending_ensemble_data():
    pred_csvs = []
    root_dir = "predictions_validation/"

    for file in os.listdir(root_dir):
        if "label" not in file:
            df = pd.read_csv(root_dir + "/" + file)
            pred_csvs.append(df)
        else:
            label_csv = pd.read_csv(root_dir + "/" + file)

    all_preds = pd.concat(pred_csvs, axis=1)
    print(all_preds)