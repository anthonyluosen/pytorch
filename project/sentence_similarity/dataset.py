import torch
import pandas as pd
from utils import config
import warnings
warnings.filterwarnings("ignore")
## data laoder
from transformers import BertTokenizer
class SentenceDataset:
    def __init__(self,df,tokenizer,max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        sentence1 = self.df.loc[item].sentence1
        sentence2 = self.df.loc[item].sentence2
        target = self.df.loc[item].label
        # Encode the tweet using the set tokenizer (converted to ids corresponding to word pieces)
        token =self.tokenizer.encode_plus(sentence1, sentence2,
                                          max_length = self.max_len,truncation=True,
                                          padding=True
                                          )
        token['target'] = target

        # padding the length
        input_ids = token['input_ids']
        mask = token['attention_mask']
        token_type_ids = token['token_type_ids']
        # truncate the sentence

        padding_length = self.max_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + ([0] * padding_length)
            mask = mask + ([0] * padding_length)
            token_type_ids = token_type_ids + ([0] * padding_length)


        return {
            'ids': torch.tensor(input_ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
            'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
            'target': target
        }

if __name__ == "__main__":
    df_valid = pd.read_csv(config.valid_path)
    valid_dataset = SentenceDataset(
        tokenizer=config.tokenizer,
        df=df_valid,
        max_len=config.max_len
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=0
    )
    for i in valid_data_loader:
        print(i)


