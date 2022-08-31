import os
import time
import sys
sys.path.append('../')

from model import TextBertClassifier, TextBertTFClassifier, TextBertweetClassifier, TextBertweetTFClassifier
from utils.evaluate import get_accuracy_from_logits, evaluate
from utils.util import log_string
from utils.dataset import TweetDataset, RawTweetDataset

import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import argparse
parser= argparse.ArgumentParser(description='ArgUtils')
parser.add_argument('-m', type=str, default='bert', help="model name")
parser.add_argument('-l', type=int, default=256, help="max length")
args = parser.parse_args()
MODEL = args.m
MAX_LEN = args.l
MODEL_DICT = {'textbert': TextBertClassifier, 'textbert_tf': TextBertTFClassifier,  'textbertweet': TextBertweetClassifier, 'textbertweet_tf': TextBertweetTFClassifier}

LOG_FOUT = open(os.path.join('./log', f'{MODEL}.txt'), 'w')
if __name__ == '__main__':
    print('read data.')
    # read the statistic data set
    train_stat = pd.read_csv('dataset/train_scaled_stat_feat_df.csv')
    # read tweet text data set
    train_data = pd.read_csv('dataset/train_tweet_df.csv')
    
    train_zero = []
    for column in train_stat.columns:
        if sum(train_stat[column] == 0) == len(train_stat):
            train_zero.append(column)
    
    # build weigthed sampler
#     class_counts = torch.tensor([len(train_data)-train_data.label.sum(), train_data.label.sum()])
#     weight = 1 / class_counts
#     weights = torch.FloatTensor([weight[i] for i in train_data.label])
    y_train = train_data['label']
    w_nonr = len(y_train)/(len(y_train)-y_train.sum())
    w_r = len(y_train)/(y_train.sum())
    weights = []
    for l in y_train:
        if l == 0:
            weights.append(w_nonr)
        else:
            weights.append(w_r)
    weights = torch.FloatTensor(weights)
    # build dataset, dataloader
    if MODEL == 'textbert' or MODEL == 'textbert_tf':
        train_set = TweetDataset('train', MAX_LEN, train_zero)
        dev_set = TweetDataset('dev', MAX_LEN, train_zero)
    else:
        train_set = RawTweetDataset('train', MAX_LEN, train_zero)
        dev_set = RawTweetDataset('dev', MAX_LEN, train_zero)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    train_sampler = Data.WeightedRandomSampler(weights, len(train_set), replacement=True)
    train_loader = Data.DataLoader(train_set, sampler=train_sampler, batch_size=64)
    dev_loader = Data.DataLoader(dev_set, batch_size=len(dev_set), shuffle=False)

    # build model
    print('load model')
    net = MODEL_DICT[MODEL]()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    # net.load_state_dict(torch.load('bertcls_0.dat'))
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    criterion = nn.BCELoss()
    opti = optim.Adam(net.parameters(), lr = 2e-5)
    print('train.')

    # training
    st = time.time()

    for ep in range(100):
        net.train()
        whole_loss = 0
        
        for it, (seq, mask, seg, stats, labels, idx) in enumerate(train_loader):
            reg_loss = 0
            for param in net.parameters():
                reg_loss += torch.sum(torch.abs(param))
            #Clear gradients
            opti.zero_grad()
            #Converting these to cuda tensors
            seq, mask, seg, stats, labels = seq.to(device), mask.to(device), seg.to(device), stats.to(device), labels.to(device)
            #Obtaining the logits from the model
            logits = net(seq, mask, seg, stats)
            #Computing loss
            loss = criterion(logits.squeeze(), labels.float()) + 0.0001 * reg_loss
            whole_loss += loss.item()
            #Backpropagating the gradients
            loss.backward()
            
            #Optimization step
            opti.step()

            if it % 10 == 0:

                acc = get_accuracy_from_logits(logits, labels)
                log_string(LOG_FOUT, "Iteration {} of epoch {} complete. \n Loss: {}; Accuracy: {}; Time taken (s): {}".format(it, ep, whole_loss / (it + 1), acc, (time.time()-st)))
                st = time.time()

        log_string(LOG_FOUT, f'epoch: {ep}, avg_loss: {whole_loss / (it+1)}')
        dev_acc, roc_auc, dev_loss = evaluate(net, criterion, dev_loader, device)
        log_string(LOG_FOUT, "Development F1: {}; Development ROCAUC: {}; Development Loss: {}".format(dev_acc, roc_auc, dev_loss))
        torch.save(net.state_dict(), f'models/{MODEL}_{ep}.dat')
