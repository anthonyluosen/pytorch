#-*-coding =utf-8 -*-
#@time :2021/12/5 10:47
#@Author: Anthony
import torch
import pandas as pd
import numpy as np
def get_predictions(loader, model, device):
    model.eval()
    save_predicted = []
    true_labels = []
    with torch.no_grad():
        for x,y in loader:
            x.to(device)
            y.to(device)
            scores = model(x)
            save_predicted += scores
            true_labels += y
    model.train()
    return save_predicted, true_labels

def get_submission(loader, model, device, test_ids):
    model.eval()
    predict_all = []
    for x, y in loader:
        x.to(device)
        # y.to(device)
        scores = model(x)
        prediction = scores.float()
        predict_all += prediction.tolist()
    model.train()

    df = pd.DataFrame(
        {
            'ID_code':test_ids.values,
            'target' :np.array(predict_all)
        }
    )
    df.to_csv('sub.csv', index=False)
