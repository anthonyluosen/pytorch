import torch
import utils as utils
import pandas as pd
import numpy as np
from tqdm.autonotebook import tqdm
from utils import config
from dataset import SentenceDataset
import transformers
from model import SentenceModel
from transformers import AdamW, get_linear_schedule_with_warmup
import utils
import warnings
warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()

def train(loss_fn,data_loader, model, optimizer,device, scheduler = None):
    losses = utils.AverageMeter()
    accuracys = utils.AverageMeter()
    model.train()
    tk0 = tqdm(data_loader, total = len(data_loader))
    for idx , d in enumerate(tk0):
        ids = d["ids"]
        token_type_ids = d["token_type_ids"]
        mask = d["mask"]
        targets = d["target"]

        # Move ids, masks, and targets to gpu while setting as torch.long
        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        model.zero_grad()
        target_predict = model(ids = ids,
                                mask = mask,
                                token_type_ids = token_type_ids)
        loss = loss_fn(target_predict.view(-1),targets)

        loss.backward()
        optimizer.step()
        scheduler.step()
        target_predict[[target_predict > 0.5]] = 1
        target_predict[[target_predict < 0.5]] = 0
        real_sum = (target_predict.view(-1) == targets).sum()
        accuracy = real_sum / ids.shape[0]
        accuracys.update(accuracy.item(),ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg,accuracy =accuracys.avg)

def eva_fn(data_loader, model, device,loss_fn):
    model.eval()
    losses = utils.AverageMeter()
    accuracys = utils.AverageMeter()

    with torch.no_grad():
        tk0 = tqdm(data_loader, total=len(data_loader))
        # Make predictions and calculate loss / jaccard score for each batch
        for bi, d in enumerate(tk0):

            ids = d["ids"]
            token_type_ids = d["token_type_ids"]
            mask = d["mask"]
            targets = d["target"]

            # Move tensors to GPU for faster matrix calculations
            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            # Predict logits for start and end indexes
            target_predict = model(ids=ids,
                                   mask=mask,
                                   token_type_ids=token_type_ids)
            loss = loss_fn(target_predict.view(-1), targets)
            target_predict[[target_predict > 0.5]] = 1
            target_predict[[target_predict < 0.5]] = 0
            real_sum = (target_predict.view(-1) == targets).sum()
            accuracy = real_sum / ids.shape[0]
            accuracys.update(accuracy.item(), ids.size(0))
            losses.update(loss.item(), ids.size(0))
            tk0.set_postfix(loss=losses.avg, valid_accuracy=accuracys.avg)
        return accuracys.avg

def run():
    dfx = pd.read_csv(config.train_path)
    df_valid = pd.read_csv(config.valid_path)
    train_dataset = SentenceDataset(
        tokenizer=config.tokenizer,
        df=dfx,
        max_len=config.max_len
    )
    valid_dataset = SentenceDataset(
        tokenizer=config.tokenizer,
        df=df_valid,
        max_len=config.max_len
    )
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=0
    )
    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=0
    )

    device = torch.device("cuda:0")
    model_config = transformers.BertConfig.from_pretrained(config.BERT_PATH,output_hidden_states=True)

    model = SentenceModel(conf=model_config,max_len=config.max_len)
    model.to(device)
    num_train_steps = int(len(dfx) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.001},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
    ]
    optimizer = AdamW(optimizer_parameters, lr=config.lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_steps
    )
    es = utils.EarlyStopping(patience=2, mode="max")
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for epoch in range(3):
        train(loss_fn,train_data_loader, model, optimizer, device, scheduler=scheduler)
        print('test data in the valid')
        accuracy = eva_fn(valid_data_loader,model,device,loss_fn)
        es(accuracy, model, model_path=f"model.bin")
        if es.early_stop:
            print("Early stopping")
            break
if __name__ == '__main__':
    run()