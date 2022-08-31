import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import DataFrame, Series
import torch
from torch.optim import lr_scheduler
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from transformers import BertModel
from transformers import BertTokenizer
import time
# import sampler as sp

class AdTweetDataset(Data.Dataset):
    def __init__(self, seq, mask, seg, source_seq, source_mask, source_seg, y):
        self.seq = torch.tensor(seq)
        self.mask = torch.tensor(mask)
        self.seg = torch.tensor(seg)
        self.source_seq = torch.tensor(source_seq)
        self.source_mask = torch.tensor(source_mask)
        self.source_seg = torch.tensor(source_seg)
        self.y = torch.tensor(y)

    def __len__(self):
        return self.seq.shape[0]

    def __getitem__(self, idx):
        return self.seq[idx],self.mask[idx],self.seg[idx], self.source_seq[idx],self.source_mask[idx],self.source_seg[idx], self.y[idx], idx

class RumorEmbedder(nn.Module):

    def __init__(self):
        super(RumorEmbedder, self).__init__()
        #Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        self.ffnn = nn.Sequential(nn.Linear(2*768, 768),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                 nn.Linear(768,512)
                                 )

    def forward(self, seq, attn_masks, seg, source_seq, attn_source_masks, source_seg):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq, attention_mask=attn_masks, token_type_ids=seg, return_dict=True)
        print('-----')
        source_outs = self.bert_layer(source_seq, attention_mask=attn_source_masks, token_type_ids=source_seg, return_dict=True)
        cont_reps = outputs.last_hidden_state
        source_cont_reps = source_outs.last_hidden_state
        #Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]
        source_cls_rep = source_cont_reps[:, 0]
        x = torch.cat((cls_rep, source_cls_rep),dim=1)
        # #Feeding cls_rep to the classifier layer
        embs = self.ffnn(x)

        return embs

def cos_sim(a,b):
    norm_a = torch.norm(a, dim=1)
    norm_b = torch.norm(b, dim=0)
    return torch.mm(a,b)/torch.mm(norm_a.unsqueeze(1), norm_b.unsqueeze(0))

class IntraInterLoss(nn.Module):

    def __init__(self):
        super(IntraInterLoss, self).__init__()

    def forward(self, emb, target):
        nr_emb = emb[target==0]
        r_emb = emb[target==1]
        count_1_1 = r_emb.shape[0]**2
        count_0_0 = nr_emb.shape[0]**2
        count_0_1 = r_emb.shape[0]*nr_emb.shape[0]
        r_cos = torch.sum(cos_sim(r_emb, r_emb.T))/float(count_1_1)
        nr_cos = torch.sum(cos_sim(nr_emb, nr_emb.T))/float(count_0_0)
        r_nr_cos = torch.sum(cos_sim(r_emb, nr_emb.T))/float(count_0_1)

        return r_nr_cos- 0.4 * r_cos- 0.1 * nr_cos

torch.manual_seed(42)

train_stat = pd.read_csv('tweepy_data/res/train_scaled_stat_feat_df.csv')
dev_stat = pd.read_csv('tweepy_data/res/dev_scaled_stat_feat_df.csv')

train_tweet = pd.read_csv('tweepy_data/res/train_tweet_df.csv')
dev_tweet = pd.read_csv('tweepy_data/res/dev_tweet_df.csv')

train_stat.drop(columns=['Unnamed: 0','label'], inplace=True)
train_stat.head()

dev_stat.drop(columns=['Unnamed: 0','label'], inplace=True)
dev_stat.head()

dev_zero = []
for column in dev_stat.columns:
    if dev_stat[column].sum() == 0:
        dev_zero.append(column)

train_zero = []
for column in train_stat.columns:
    if train_stat[column].sum() == 0:
        train_zero.append(column)

train_stat.drop(columns=train_zero, inplace=True)
dev_stat.drop(columns=dev_zero, inplace=True)

dev_tweet.reply_text.fillna('', inplace=True)
train_tweet.reply_text.fillna('', inplace=True)
dev_tweet.text.fillna('', inplace=True)
train_tweet.text.fillna('', inplace=True)
dev_tweet['source'] = ''
train_tweet['source'] = ''
for i in tqdm(range(len(dev_tweet))):
    dev_tweet.source.iloc[i] = '[CLS] ' + str(dev_tweet.text.iloc[i]).strip() + ' [SEP]'
    dev_tweet.text.iloc[i] = '[CLS] ' + str(dev_tweet.text.iloc[i]).strip() + ' [SEP] ' + str(dev_tweet.reply_text.iloc[i]).strip() + ' [SEP]'

for i in tqdm(range(len(train_tweet))):
    train_tweet.source.iloc[i] = '[CLS] ' + str(train_tweet.text.iloc[i]).strip() + ' [SEP]'
    train_tweet.text.iloc[i] = '[CLS] ' + str(train_tweet.text.iloc[i]).strip() + ' [SEP] ' + str(train_tweet.reply_text.iloc[i]).strip() + ' [SEP]'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

max_len = 256
dev_seq = []
train_seq = []
dev_source_seq = []
train_source_seq = []
dev_mask = []
train_mask = []
dev_source_mask = []
train_source_mask = []
dev_seg = []
train_seg = []
dev_source_seg = []
train_source_seg = []
for i in tqdm(range(len(dev_tweet))):
    source_tokens = tokenizer.tokenize(dev_tweet.source.iloc[i])
    txt = dev_tweet.text.iloc[i]
    tokens = tokenizer.tokenize(txt)
    if len(source_tokens) < max_len:
        source_padded_tokens = source_tokens + ['[PAD]' for _ in range(max_len - len(source_tokens))]
    else:
        source_padded_tokens = source_tokens[:max_len-1] + ['[SEP]']
    if len(tokens) < max_len:
        padded_tokens = tokens + ['[PAD]' for _ in range(max_len - len(tokens))]
    else:
        padded_tokens = tokens[:max_len-1] + ['[SEP]']
    attn_source_mask = [1 if token != '[PAD]' else 0 for token in source_padded_tokens]
    attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
    seg_ids = source_seg_ids = []
    seg_idx = source_seg_idx = 0
    for token in padded_tokens:
        seg_ids.append(seg_idx)
        if token == '[SEP]':
            if seg_idx == 1:
                seg_idx = 0
            else:
                seg_idx=1
    for token in source_padded_tokens:
        source_seg_ids.append(source_seg_idx)
        if token == '[SEP]':
            if source_seg_idx == 1:
                source_seg_idx = 0
            else:
                source_seg_idx=1

    # seg_ids = [1 if token == '[SEP]' else 0 for token in padded_tokens]
    token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
    source_token_ids = tokenizer.convert_tokens_to_ids(source_padded_tokens)
    dev_seq.append(token_ids)
    dev_mask.append(attn_mask)
    dev_seg.append(seg_ids)
    dev_source_seq.append(source_token_ids)
    dev_source_mask.append(attn_source_mask)
    dev_source_seg.append(source_seg_ids)

for i in tqdm(range(len(train_tweet))):
    source_tokens = tokenizer.tokenize(train_tweet.source.iloc[i])
    txt = train_tweet.text.iloc[i]
    tokens = tokenizer.tokenize(txt)
    if len(source_tokens) < max_len:
        source_padded_tokens = source_tokens + ['[PAD]' for _ in range(max_len - len(source_tokens))]
    else:
        source_padded_tokens = source_tokens[:max_len-1] + ['[SEP]']
    if len(tokens) < max_len:

        padded_tokens = tokens + ['[PAD]' for _ in range(max_len - len(tokens))]
    else:

        padded_tokens = tokens[:max_len-1] + ['[SEP]']
    attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
    attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
    seg_ids = []
    source_seg_ids = []
    seg_idx = 0
    source_seg_idx = 0
    for token in padded_tokens:
        seg_ids.append(seg_idx)
        if token == '[SEP]':
            if seg_idx == 1:
                seg_idx = 0
            else:
                seg_idx=1
    for token in source_padded_tokens:
        source_seg_ids.append(source_seg_idx)
        if token == '[SEP]':
            if source_seg_idx == 1:
                source_seg_idx = 0
            else:
                source_seg_idx=1
    token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)
    source_token_ids = tokenizer.convert_tokens_to_ids(source_padded_tokens)
    train_seq.append(token_ids)
    train_mask.append(attn_mask)
    train_seg.append(seg_ids)
    train_source_seq.append(source_token_ids)
    train_source_mask.append(attn_source_mask)
    train_source_seg.append(source_seg_ids)

print('data loaded.')
y_train = train_tweet['label']
y_dev = dev_tweet['label']
w_nonr = len(y_train)/(len(y_train)-y_train.sum())
w_r = len(y_train)/(y_train.sum())
weights = []
for l in y_train:
    if l == 0:
        weights.append(w_nonr)
    else:
        weights.append(w_r)
weights = torch.FloatTensor(weights)
print(len(train_source_seq))
train_set = AdTweetDataset(train_seq, train_mask, train_seg,
                            train_source_seq, train_source_mask, train_source_seg, y_train)
dev_set = AdTweetDataset(dev_seq, dev_mask, dev_seg,
                        dev_source_seq, dev_source_mask, dev_source_seg, y_dev)

# sampler_s = sp.StratifiedSampler(class_vector=torch.from_numpy(np.array(y_train)), batch_size=64)
train_sampler = Data.WeightedRandomSampler(weights, len(train_set), replacement=True)
train_loader = Data.DataLoader(train_set, sampler=train_sampler,batch_size=64)
dev_loader = Data.DataLoader(dev_set, batch_size=64, shuffle=False)

dev_loader = Data.DataLoader(dev_set, batch_size=len(dev_set), shuffle=False)
# torch.cuda.empty_cache ()
net = RumorEmbedder()
# net = net.to(device)

criterion = IntraInterLoss()
opti = optim.Adam(net.parameters(), lr = 0.001)
scheduler = lr_scheduler.ExponentialLR(opti, gamma=0.7)
st = time.time()
print('begin training.')
for ep in range(20):
    net.train()
    for it, (seq, mask, seg, source_seq, source_mask, source_seg, labels,idx) in enumerate(train_loader):
        print(f'epoch: {ep}, it: {it}.==================')
        #Clear gradients
        opti.zero_grad()
        #Converting these to cuda tensors
        # seq, mask, seg, labels = seq.to(device), mask.to(device), seg.to(device), labels.to(device)

        stats = np.array(train_stat)
        stats = torch.tensor(stats[idx]).float()
        #Obtaining the logits from the model
        logits = net(seq, mask, seg, source_seq, source_mask, source_seg)
        #Computing loss
        loss = criterion(logits, labels)

        #Backpropagating the gradients
        loss.backward()

        #Optimization step
        opti.step()

        if it % 10 == 0:

            print("Iteration {} of epoch {} complete. \n Loss: {}; Time taken (s): {}".format(it, ep, loss.item(), (time.time()-st)))
            st = time.time()
    scheduler.step()
    net.eval()
    with torch.no_grad():
        for seq, mask, seg, source_seq, source_mask, source_seg, labels, idx in dev_loader:
            dev_embs = net(seq, mask, seg, source_seq, source_mask, source_seg)
        dev_loss = criterion(dev_embs, labels)

    print("Development Loss: {}".format(dev_loss.item()))
    torch.save({
                  'model': net.state_dict(),
                #   'optimizer': opti.state_dict(),
                #   'scheduler': scheduler.state_dict(),
                #   'iteration': ep
                }, f'./models/bertemb_{ep}.dat')
