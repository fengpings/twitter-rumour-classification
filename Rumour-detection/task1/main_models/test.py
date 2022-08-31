import pandas as pd
from tqdm import tqdm

import torch
import torch.utils.data as Data

from utils.dataset import TweetDataset, RawTweetDataset
from model import TextBertClassifier, TextBertTFClassifier, TextBertweetClassifier, TextBertweetTFClassifier
import argparse
parser= argparse.ArgumentParser(description='ArgUtils')
parser.add_argument('-m', type=str, default='bert', help="model name")
parser.add_argument('-l', type=int, default=256, help="max length")
parser.add_argument('-t', type=str, default='test', help="test data name, test/covid")
parser.add_argument('-e', type=str, default='bertcls_16.dat', help="the path of existing model")

args = parser.parse_args()
MODEL = args.m
MAX_LEN = args.l
TYPE = args.t
EXIST_MODEL = args.e

MODEL_DICT = {'textbert': TextBertClassifier, 'textbert_tf': TextBertTFClassifier,  'textbertweet': TextBertweetClassifier, 'textbertweet_tf': TextBertweetTFClassifier}

if __name__ == '__main__':
    train_stat = pd.read_csv('dataset/train_scaled_stat_feat_df.csv')
    train_stat.drop(columns=['Unnamed: 0', 'label'], inplace=True)
    train_zero = []
    for column in train_stat.columns:
        if sum(train_stat[column] == 0) == len(train_stat):
            train_zero.append(column)
    if MODEL == 'textbert' or MODEL == 'textbert_tf':         
        test_set = TweetDataset(TYPE, MAX_LEN, train_zero)
    else:
        test_set = RawTweetDataset(TYPE, MAX_LEN, train_zero)
    test_loader = Data.DataLoader(test_set, batch_size=64, shuffle=False, sampler=range(0,len(test_set)))
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    model = MODEL_DICT[MODEL]()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load(EXIST_MODEL))
    model.eval()
    preds = []

    with torch.no_grad():
        for i, (seq, mask, seg, stat, idx) in tqdm(enumerate(test_loader)):
            preds.append(model(seq.to(device), mask.to(device), seg.to(device), stat.to(device)))

    predictions = []
    for i in range(len(preds)):
        cur_df = list(preds[i].cpu().numpy())
        predictions.extend(cur_df)

    pred_dict = {'Id':[i for i in range(len(predictions))], 'Predicted':predictions}
    pred_df = pd.DataFrame(pred_dict)

    pred_df.Predicted = pred_df.Predicted.apply(lambda x: 1 if x > 0.5 else 0)
    pred_df.to_csv(f'res/predictions_{MODEL}_{TYPE}.csv', index=False)
    print('save')