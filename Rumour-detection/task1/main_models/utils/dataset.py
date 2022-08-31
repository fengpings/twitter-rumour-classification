import torch
import torch.utils.data as Data

from transformers import BertTokenizer, BertModel
# from transformers import AutoTokenizer
from transformers import AutoTokenizer
import transformers
transformers.logging.set_verbosity_error()
import pandas as pd

class TweetDataset(Data.Dataset):
    def __init__(self, data_type, max_len, zero_cols):
        tweet_df = pd.read_csv(f'dataset/{data_type}_tweet_df.csv')
        static_df = pd.read_csv(f'dataset/{data_type}_scaled_stat_feat_df.csv')
        if data_type == 'train' or data_type == 'dev':
            static_df.drop(columns=['Unnamed: 0', 'label'], inplace=True)
        else:
            static_df.drop(columns=['tweet_id'], inplace=True)
        static_df.drop(columns=zero_cols, inplace=True)
        static_df.fillna('', inplace=True)
        tweet_df.reply_text.fillna('', inplace=True)
        tweet_df.text.fillna('', inplace=True)
        tweet_df.text = tweet_df.apply(lambda x: '[CLS] ' + str(x['text']).strip() + ' [SEP] ' + str(x['reply_text']).strip() + ' [SEP]', axis=1)
        self.data_type = data_type
        self.tweet_df = tweet_df
        self.static_df = static_df
        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    def __len__(self):
        return len(self.tweet_df)
    def __getitem__(self, idx):
        txt = self.tweet_df.text.iloc[idx]
        static = self.static_df.iloc[idx]
        if self.data_type != 'test':
            label = self.tweet_df.label.iloc[idx]
        tokens = self.tokenizer.tokenize(txt)
        if len(tokens) < self.max_len:
            padded_tokens = tokens + ['[PAD]' for _ in range(self.max_len - len(tokens))]
        else:
            padded_tokens = tokens[: self.max_len - 1] + ['[SEP]']
        attn_mask = [0 if i == '[PAD]' else 1 for i in padded_tokens]
        seg_index = 0
        seg_id = []
        for i in padded_tokens:
            seg_id.append(seg_index)
            if i == '[SEP]':
                if seg_index == 1:
                    seg_index = 0
                else:
                    seg_index = 1
        token_ids = self.tokenizer.convert_tokens_to_ids(padded_tokens)
        if self.data_type == 'train' or self.data_type == 'dev':
            return torch.tensor(token_ids), torch.tensor(attn_mask), torch.tensor(seg_id), torch.Tensor(static), torch.tensor(label), idx
        else:
            return torch.tensor(token_ids), torch.tensor(attn_mask), torch.tensor(seg_id), torch.Tensor(static), idx

class AdTweetDataset(Data.Dataset):
    def __init__(self, seq_, mask_, seg_, y_):
        self.seq = torch.tensor(seq_)
        self.mask = torch.tensor(mask_)
        self.seg = torch.tensor(seg_)
        self.y = torch.tensor(y_)

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx):
        return self.seq[idx],self.mask[idx],self.seg[idx], self.y[idx], idx

class RawTweetDataset(Data.Dataset):
    def __init__(self, data_type, max_len, zero_cols):
        tweet_df = pd.read_csv(f'dataset/raw/{data_type}_tweet_df.csv')
        static_df = pd.read_csv(f'dataset/{data_type}_scaled_stat_feat_df.csv')
        if data_type == 'train' or data_type == 'dev':
            static_df.drop(columns=['Unnamed: 0', 'label'], inplace=True)
        else:
            static_df.drop(columns=['tweet_id'], inplace=True)
        static_df.drop(columns=zero_cols, inplace=True)
        static_df.fillna('', inplace=True)
        tweet_df.reply_text.fillna('', inplace=True)
        tweet_df.text.fillna('', inplace=True)
        self.data_type = data_type
        self.tweet_df = tweet_df
        self.static_df = static_df
        self.max_len = max_len
        self.tokenizer =  AutoTokenizer.from_pretrained("vinai/bertweet-base",cache_dir='./bertweet_base',local_files_only=True, normalization=True)
        
    def __len__(self):
        return len(self.tweet_df)
    def __getitem__(self, idx):
        static = self.static_df.iloc[idx]
        if self.data_type == 'train' or self.data_type == 'dev':
            label = self.tweet_df.label.iloc[idx]
        tokens_mask = self.tokenizer(self.tweet_df.text.iloc[idx], self.tweet_df.reply_text.iloc[idx], truncation=True, padding='max_length', max_length=self.max_len)
        token_ids, attn_mask = tokens_mask['input_ids'], tokens_mask['attention_mask']
        if self.data_type == 'train' or self.data_type == 'dev':
            return torch.tensor(token_ids), torch.tensor(attn_mask), torch.zeros(len(attn_mask)) ,torch.Tensor(static),  torch.tensor(label), idx
        else:
            return torch.tensor(token_ids), torch.tensor(attn_mask),torch.zeros(len(attn_mask)) , torch.Tensor(static),  idx
class TreeDataset(Data.Dataset):
    def __init__(self, data_type, max_len, tokenizer, bert_layer):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tweet_df = pd.read_csv(f'dataset/{data_type}_tree.csv')
        tweet_df.text.fillna('', inplace=True)
        tweet_df.text = tweet_df['text'].apply(lambda x: '[CLS] ' + str(x).strip() + ' [SEP]')
        self.data_type = data_type
        self.tweet_df = tweet_df
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.bert_layer = bert_layer.to(device)
        self.device = device
    def __len__(self):
        return len(self.tweet_df)
    def __getitem__(self, idx):
        tweet_id = self.tweet_df.tweet_id.iloc[idx]
        txt = self.tweet_df.text.iloc[idx]
        tokens = self.tokenizer.tokenize(txt)
        if len(tokens) < self.max_len:
            padded_tokens = tokens + ['[PAD]' for _ in range(self.max_len - len(tokens))]
        else:
            padded_tokens = tokens[: self.max_len - 1] + ['[SEP]']
        attn_mask = [0 if i == '[PAD]' else 1 for i in padded_tokens]
        seg_index = 0
        seg_id = []
        for i in padded_tokens:
            seg_id.append(seg_index)
            if i == '[SEP]':
                if seg_index == 1:
                    seg_index = 0
                else:
                    seg_index = 1
        token_ids = self.tokenizer.convert_tokens_to_ids(padded_tokens)
        emb = self.bert_layer(torch.tensor(token_ids).unsqueeze(0).to(self.device), attention_mask=torch.tensor(attn_mask).unsqueeze(0).to(self.device), token_type_ids=torch.tensor(seg_id).unsqueeze(0).to(self.device)).last_hidden_state[:,0]
        return emb, tweet_id
