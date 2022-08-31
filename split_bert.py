import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import DataFrame, Series
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.utils.data as Data
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from transformers import BertModel
from transformers import BertTokenizer
import lightgbm as lgb
import xgboost as xgb
import optuna
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE,KMeansSMOTE, SVMSMOTE
from imblearn.combine import SMOTEENN
from sklearn.metrics import roc_auc_score
import pickle
import time

def generate_seq(tweet, tokenizer, max_len):
    seq = []
    mask = []
    seg = []
    for i in tqdm(range(len(tweet))):
        txt = tweet.iloc[i]
        tokens = tokenizer.tokenize(txt)
        if len(tokens) < max_len:
             padded_tokens = tokens + ['[PAD]' for _ in range(max_len - len(tokens))]
        else:
            padded_tokens = tokens[:max_len-1] + ['[SEP]']

        attn_mask = [1 if token != '[PAD]' else 0 for token in padded_tokens]
        seg_ids = []
        seg_idx = 0
        for token in padded_tokens:
            seg_ids.append(seg_idx)
            if token == '[SEP]':
                if seg_idx == 1:
                    seg_idx = 0
                else:
                    seg_idx=1
        token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)

        seq.append(token_ids)
        mask.append(attn_mask)
        seg.append(seg_ids)
    return seq, mask, seg

def get_accuracy_from_logits(logits, labels):
    probs = logits.unsqueeze(-1)
    soft_probs = (probs > 0.5).long()
    acc = (soft_probs.squeeze() == labels).float().mean()
    return acc

def get_f1_from_logits(logits, labels):
    preds = (logits >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(labels, preds, pos_label=1, average="binary")
    return f

def get_roc_auc_from_logits(logits, labels):
    preds = (logits >= 0.5).astype(int)
    return roc_auc_score(preds,labels)

def evaluate(net, criterion, dataloader, device):
    net.eval()

    mean_acc, mean_loss = 0, 0
    count = 0
    all_log = np.array([])
    all_labels = np.array([])
    with torch.no_grad():
        for src_seq, src_mask, src_seg, rp_seq, rp_mask, rp_seg, labels, idx in dataloader:
            # seq, labels = seq.to(device), labels.to(device)
            stats = np.array(dev_stat)
            stats = torch.tensor(stats[idx]).float()
            #Obtaining the logits from the model
            logits = net(src_seq, src_mask, rp_seq, rp_mask, rp_seg, src_seg, stats)
            mean_loss += criterion(logits.squeeze(-1), labels.float()).item()
            # mean_acc += get_accuracy_from_logits(logits, labels)

            all_log = np.hstack((all_log, logits.squeeze()))
            all_labels = np.hstack((all_labels, labels.numpy()))
            count += 1

        f = get_f1_from_logits(all_log, all_labels)
        roc_auc = get_roc_auc_from_logits(all_log, all_labels)
    return f, roc_auc, mean_loss / count

class RumorClassifier(nn.Module):

    def __init__(self):
        super(RumorClassifier, self).__init__()
        #Instantiating BERT model object
        self.src_layer = BertModel.from_pretrained('bert-base-uncased')
        self.rp_layer = BertModel.from_pretrained('bert-base-uncased')

        self.ffnn = nn.Sequential(nn.Linear(1579,512),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Linear(512,128),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                 nn.Linear(128,64),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Linear(64,1),
                                  nn.Sigmoid()
                                 )

    def forward(self, src_seq, src_mask, src_seg, rp_seq, rp_mask, rp_seg, stats):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        src_outputs = self.src_layer(src_seq, attention_mask=src_mask,token_type_ids=src_seg, return_dict=True)
        rp_outputs = self.rp_layer(rp_seq, attention_mask=rp_mask,token_type_ids=rp_seg, return_dict=True)
        src_cont_reps = src_outputs.last_hidden_state
        rp_cont_reps = rp_outputs.last_hidden_state

        #Obtaining the representation of [CLS] head (the first token)
        src_cls_rep = src_cont_reps[:, 0]
        rp_cls_rep = rp_cont_reps[:,0]

        x = torch.cat((src_cls_rep,rp_cls_rep), dim=1)
        x = torch.cat((x,stats),dim=1)
        #Feeding cls_rep to the classifier layer
        logits = self.ffnn(x)

        return logits

class AdTweetDataset(Data.Dataset):
    def __init__(self, src_seq, src_mask, src_seg, rp_seq, rp_mask, rp_seg, y):
        self.src_seq = torch.tensor(src_seq)
        self.src_mask = torch.tensor(src_mask)
        self.src_seg = torch.tensor(src_seg)
        self.rp_seq = torch.tensor(rp_seq)
        self.rp_mask = torch.tensor(rp_mask)
        self.rp_seg = torch.tensor(rp_seg)
        self.y = torch.tensor(y)

    def __len__(self):
        return self.src_seq.shape[0]

    def __getitem__(self, idx):
        return self.src_seq[idx],self.src_mask[idx],self.src_seg[idx], self.rp_seq[idx],self.rp_mask[idx],self.rp_seg[idx],self.y[idx], idx

train_stat = pd.read_csv('./sep_data/train_scaled_stat_feat_df.csv')
dev_stat = pd.read_csv('./sep_data/dev_scaled_stat_feat_df.csv')

train_tweet = pd.read_csv('./sep_data/train_tweet_df.csv')
dev_tweet = pd.read_csv('./sep_data/dev_tweet_df.csv')

# train_stat.info()

# dev_stat.info()

train_stat.drop(columns=['Unnamed: 0','label'], inplace=True)
train_stat.head()

dev_stat.drop(columns=['Unnamed: 0','label'], inplace=True)
dev_stat.head()

train_stat.sum()

dev_zero = []
for column in dev_stat.columns:
    if (dev_stat[column] != 0).sum() == 0:
        dev_zero.append(column)
print(dev_zero)

train_zero = []
for column in train_stat.columns:
    if (train_stat[column] != 0).sum() == 0:
        train_zero.append(column)
print(train_zero)

train_stat.drop(columns=train_zero, inplace=True)
dev_stat.drop(columns=dev_zero, inplace=True)

dev_tweet.head()

dev_tweet.reply_text.fillna('None', inplace=True)
train_tweet.reply_text.fillna('None', inplace=True)
dev_tweet.text.fillna('None', inplace=True)
train_tweet.text.fillna('None', inplace=True)

for i in tqdm(range(len(dev_tweet))):
    dev_tweet.text.iloc[i] = '[CLS] ' + str(dev_tweet.text.iloc[i]).strip() + ' [SEP]'
    dev_tweet.reply_text.iloc[i] = '[CLS] ' + str(dev_tweet.reply_text.iloc[i]).strip() + ' [SEP]'

for i in tqdm(range(len(train_tweet))):
    train_tweet.text.iloc[i] = '[CLS] ' + str(train_tweet.text.iloc[i]).strip() + ' [SEP]'
    train_tweet.reply_text.iloc[i] = '[CLS] ' + str(train_tweet.reply_text.iloc[i]).strip() + ' [SEP]'

train_src_seq, train_src_mask, train_src_seg = generate_seq(train_tweet.text, tokenizer, 256)
train_rp_seq, train_rp_mask, train_rp_seg = generate_seq(train_tweet.reply_text, tokenizer, 512)
dev_src_seq, dev_src_mask, dev_src_seg = generate_seq(dev_tweet.text, tokenizer,256)
dev_rp_seq, dev_rp_mask, dev_rp_seg = generate_seq(dev_tweet.reply_text, tokenizer,512)

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

torch.manual_seed(42)
train_set = AdTweetDataset(train_src_seq, train_src_mask, train_src_seg,train_rp_seq, train_rp_mask, train_rp_seg,y_train)
dev_set = AdTweetDataset(dev_src_seq, dev_src_mask, dev_src_seg,dev_rp_seq, dev_rp_mask, dev_rp_seg,y_train)

# sampler_s = sp.StratifiedSampler(class_vector=torch.from_numpy(np.array(y_train)), batch_size=64)
train_sampler = Data.WeightedRandomSampler(weights, len(train_set), replacement=True)
train_loader = Data.DataLoader(train_set, sampler=train_sampler,batch_size=16)
# train_loader = Data.DataLoader(train_set,batch_size=64,shuffle=True)
dev_loader = Data.DataLoader(dev_set, batch_size=16, shuffle=False)

net = RumorClassifier()
# net = net.to(device)

criterion = nn.BCELoss()
opti = optim.Adam(net.parameters(), lr = 2e-5)

best_acc = 0
st = time.time()
eps = []
t_loss = []
d_loss = []

for ep in range(20):
    eps.append(ep)
    net.train()
    for it, (src_seq, src_mask, src_seg, rp_seq, rp_mask, rp_seg, labels, idx) in enumerate(train_loader):

        #Clear gradients
        opti.zero_grad()
        #Converting these to cuda tensors
        # seq, mask, seg, labels = seq.to(device), mask.to(device), seg.to(device), labels.to(device)

        stats = np.array(train_stat)
        stats = torch.tensor(stats[idx]).float()
        #Obtaining the logits from the model
        logits = net(src_seq, src_mask, src_seg, rp_seq, rp_mask, rp_seg, stats)
        #Computing loss
        loss = criterion(logits.squeeze(), labels.float())

        #Backpropagating the gradients
        loss.backward()

        #Optimization step
        opti.step()

        if it % 10 == 0:

            acc = get_accuracy_from_logits(logits, labels)
            print("Iteration {} of epoch {} complete. \n Loss: {}; Accuracy: {}; Time taken (s): {}".format(it, ep, loss.item(), acc, (time.time()-st)))
            st = time.time()


    dev_acc, roc_auc, dev_loss = evaluate(net, criterion, dev_loader, 'cpu')
    t_loss.append(loss.item())
    d_loss.append(dev_loss)
    print("Development F1: {}; Development ROCAUC: {}; Development Loss: {}".format(dev_acc, roc_auc, dev_loss))
    torch.save(net.state_dict(), 'D:\\bertcls_{}.dat'.format(ep))
