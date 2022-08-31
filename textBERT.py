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

class TextRumorClassifier(nn.Module):

    def __init__(self):
        super(TextRumorClassifier, self).__init__()
        #Instantiating BERT model object
        self.bert_layer = BertModel.from_pretrained('bert-base-uncased')

        self.ffnn = nn.Sequential(nn.Linear(768,128),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                 nn.Linear(128,64),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Linear(64,1),
                                  nn.Sigmoid()
                                 )
        # self.metric_fc = ArcMarginProduct(768, 2)

    def forward(self, seq, attn_masks, seg):
        '''
        Inputs:
            -seq : Tensor of shape [B, T] containing token ids of sequences
            -attn_masks : Tensor of shape [B, T] containing attention masks to be used to avoid contibution of PAD tokens
        '''

        #Feeding the input to BERT model to obtain contextualized representations
        outputs = self.bert_layer(seq, attention_mask = attn_masks, return_dict=True)
        cont_reps = outputs.last_hidden_state

        #Obtaining the representation of [CLS] head (the first token)
        cls_rep = cont_reps[:, 0]

        # x = torch.cat((cls_rep,stats),dim=1)
        #Feeding cls_rep to the classifier layer
        logits = self.ffnn(cls_rep)
        # output = self.metric_fc(cls_rep, labels)

        return logits
        # return output
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
        for i, (seq, mask, seg, labels,idx) in enumerate(dataloader):
            # stats = np.array(dev_stat)
            # stats = torch.tensor(stats[idx]).float()
            #Obtaining the logits from the model
            logits = net(seq, mask, seg)
            mean_loss += criterion(logits.squeeze(-1), labels.long()).item()
            # mean_acc += get_accuracy_from_logits(logits, labels)
            # print()
            # print(logits.squeeze().shape)
            if i == 0:
                all_log = logits.squeeze().numpy()
                # print(all_log.shape)
            else:
                all_log = np.vstack((all_log, logits.squeeze().numpy()))
            all_labels = np.hstack((all_labels, labels.numpy()))
            count += 1

        f = get_f1_from_logits(all_log, all_labels)
        roc_auc = get_roc_auc_from_logits(all_log, all_labels)
    return f, roc_auc, mean_loss / count

train_stat = pd.read_csv('./sep_data/train_scaled_stat_feat_df.csv')
dev_stat = pd.read_csv('./sep_data/dev_scaled_stat_feat_df.csv')

train_tweet = pd.read_csv('./sep_data/train_tweet_df.csv')
dev_tweet = pd.read_csv('./sep_data/dev_tweet_df.csv')

# train_stat.info()

# dev_stat.info()

train_stat.drop(columns=['Unnamed: 0','label'], inplace=True)
dev_stat.drop(columns=['Unnamed: 0','label'], inplace=True)

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


dev_tweet.reply_text.fillna('', inplace=True)
train_tweet.reply_text.fillna('', inplace=True)
dev_tweet.text.fillna('', inplace=True)
train_tweet.text.fillna('', inplace=True)

for i in tqdm(range(len(dev_tweet))):
    dev_tweet.text.iloc[i] = '[CLS] ' + str(dev_tweet.text.iloc[i]).strip() + ' [SEP] ' + str(dev_tweet.reply_text.iloc[i]).strip() + ' [SEP]'

for i in tqdm(range(len(train_tweet))):
    train_tweet.text.iloc[i] = '[CLS] ' + str(train_tweet.text.iloc[i]).strip() + ' [SEP] ' + str(train_tweet.reply_text.iloc[i]).strip() + ' [SEP]'

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

max_len = 256
dev_seq = []
train_seq = []
dev_mask = []
train_mask = []
dev_seg = []
train_seg = []

for i in tqdm(range(len(dev_tweet))):
    txt = dev_tweet.text.iloc[i]
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
    # seg_ids = [1 if token == '[SEP]' else 0 for token in padded_tokens]
    token_ids = tokenizer.convert_tokens_to_ids(padded_tokens)

    dev_seq.append(token_ids)
    dev_mask.append(attn_mask)
    dev_seg.append(seg_ids)

for i in tqdm(range(len(train_tweet))):
    txt = train_tweet.text.iloc[i]
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

    train_seq.append(token_ids)
    train_mask.append(attn_mask)
    train_seg.append(seg_ids)

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
train_set = AdTweetDataset(train_seq, train_mask, train_seg,y_train)
dev_set = AdTweetDataset(dev_seq, dev_mask, dev_seg, y_dev)

# sampler_s = sp.StratifiedSampler(class_vector=torch.from_numpy(np.array(y_train)), batch_size=64)
train_sampler = Data.WeightedRandomSampler(weights, len(train_set), replacement=True)
train_loader = Data.DataLoader(train_set, sampler=train_sampler,batch_size=64)
# train_loader = Data.DataLoader(train_set,batch_size=64,shuffle=True)
dev_loader = Data.DataLoader(dev_set, batch_size=64, shuffle=False)

# torch.cuda.empty_cache ()
net = TextRumorClassifier()
# net = net.to(device)

# criterion = nn.BCELoss()
criterion=torch.nn.BCELoss()
# scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
opti = optim.Adam(net.parameters(), lr = 2e-5)

best_acc = 0
st = time.time()
eps = []
t_loss = []
d_loss = []

for ep in range(100):
    net.train()
    for it, (seq, mask, seg, labels,idx) in enumerate(train_loader):

        #Clear gradients
        opti.zero_grad()
        # print(x)
        # x = torch.tensor(x).float()

        # stats = np.array(train_stat)
        # stats = torch.tensor(stats[idx]).float()
        #Obtaining the logits from the model
        logits = net(seq, mask, seg)
        #Computing loss
        loss = criterion(logits.squeeze(), labels.long())

        #Backpropagating the gradients
        loss.backward()

        #Optimization step
        opti.step()

        if it % 10 == 0:

            acc = get_accuracy_from_logits(logits, labels)
            print("Iteration {} of epoch {} complete. \n Loss: {}; Accuracy: {}; Time taken (s): {}".format(it, ep, loss.item(), acc, (time.time()-st)))
            st = time.time()


    dev_acc, roc_auc, dev_loss = evaluate(net, criterion, dev_loader, 'cpu')
    print("Development F1: {}; Development ROCAUC: {}; Development Loss: {}".format(dev_acc, roc_auc, dev_loss))
    torch.save(net.state_dict(), 'D:\\bertcls_{}.dat'.format(ep))
