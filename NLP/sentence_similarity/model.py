import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from utils import config

class SentenceModel(transformers.BertPreTrainedModel):
    def __init__(self,conf,max_len):
        super(SentenceModel,self).__init__(conf)
        self.bert = transformers.BertModel.from_pretrained(config.bert_path,config = conf
                                                           )
        self.drop = nn.Dropout(0.1)
        self.linear0 = nn.Linear(2*768,1)
        self.linear1 = nn.Linear(max_len, 1)
        torch.nn.init.normal_(self.linear0.weight,std=0.02)
    def forward(self,ids,mask,token_type_ids):
        out = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        )
        #here we get the dimension [num of hiddenstate , batch_size,sentence_len,768]
        out = torch.cat((out.hidden_states[-1],out.hidden_states[-2]),dim=-1)
        # so here we get  [batch size , sentence len , 2*768]
        out = self.drop(out)
        out = self.linear0(out)
        output = out.view(ids.shape[0], -1)
        # output shape[batch size ,1]
        return self.linear1(output)

