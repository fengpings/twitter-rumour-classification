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

X_train = pd.read_csv('./embs/X_train.csv').iloc[:,1:]
X_dev = pd.read_csv('./embs/X_dev.csv').iloc[:,1:]
y_train = pd.read_csv('./embs/y_train.csv')['label']
y_dev = pd.read_csv('./embs/y_dev.csv')['label']

train_stat = pd.read_csv('./user_rumor/train_scaled_stat_feat_df.csv')
dev_stat = pd.read_csv('./user_rumor/dev_scaled_stat_feat_df.csv')
y_train = train_stat.label
y_dev = dev_stat.label
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

X_train = pd.concat([X_train.iloc[:,:768],train_stat],axis=1)
X_dev = pd.concat([X_dev.iloc[:,:768],dev_stat],axis=1)

# from sklearn.decomposition import PCA
# pca_mle = PCA(n_components=0.999)
# pca_mle = pca_mle.fit(X_train)
# X_train = pca_mle.transform(X_train)
# X_dev = pca_mle.transform(X_dev)
#
# from sklearn.preprocessing import PolynomialFeatures
# poly = PolynomialFeatures(2)
# X_train = poly.fit_transform(X_train)
# X_dev = poly.transform(X_dev)

# sm = SVMSMOTE()
# X_train,y_train = sm.fit_resample(X_train, y_train)

from sklearn.model_selection import train_test_split
# train,val,y,y_val = train_test_split(X_train,
#                                         y_train,
#                                         train_size=0.66,
#                                         stratify=y_train,
#                                         random_state=42)
def gbm_objective(trial):
    n_estimators = trial.suggest_int('n_estimators',1,2001,1)
    learning_rate = trial.suggest_loguniform('learning_rate',1e-5,1.0)
    max_depth = trial.suggest_int('max_depth',2,101,1)
    num_leaves = trial.suggest_int('num_leaves',2,101,1)
    subsample_freq = trial.suggest_int('subsample_freq',2,101,1)
    min_child_samples = trial.suggest_int('min_child_samples',2,101,1)
    min_child_weight = trial.suggest_float('min_child_weight',0.0001,1.0,log=False)
    colsample_bytree = trial.suggest_float('colsample_bytree',0.5,1.0,log=False)
    subsample = trial.suggest_float('subsample',0.5,1.0,log=False)
    reg_alpha = trial.suggest_float('reg_alpha',0.0,1.0,log=False)
    reg_lambda = trial.suggest_float('reg_lambda',0.5,1.0,log=False)
    min_split_gain = trial.suggest_float('min_split_gain',0.0,1.0,log=False)
    max_bin = trial.suggest_int('max_bin',20,512,1)



    classifier_obj = lgb.LGBMClassifier(objective = 'binary',
                                       n_estimators=n_estimators,
                                       learning_rate=learning_rate,
                                       max_depth=max_depth,
                                        num_leaves=num_leaves,
                                       min_child_samples=min_child_samples,
                                        min_child_weight=min_child_weight,
                                       colsample_bytree=colsample_bytree,
                                       subsample=subsample,
                                        subsample_freq=subsample_freq,
                                       reg_alpha=reg_alpha,
                                       reg_lambda=reg_lambda,
                                       min_split_gain=min_split_gain,
                                       max_bin=max_bin,
                                       random_state=42)

    classifier_obj.fit(X_train,y_train)
    predictions = classifier_obj.predict(X_dev).astype(int)
    if np.sum(predictions) == 0:
        predictions[0] = 1
    # return roc_auc_score(predictions,y_dev)
    p, r, f, _ = precision_recall_fscore_support(y_dev, predictions, pos_label=1, average="binary")
    return f
    # return roc_auc_score(predictions,y_dev)

study = optuna.create_study(direction='maximize')
study.optimize(gbm_objective, n_trials=500,n_jobs=-1)
print(study.best_params)
print(study.best_value)
print(study.best_trial)

params = study.best_params
gbm = lgb.LGBMClassifier(**params,random_state=42,objective='binary')
gbm.fit(X_train,y_train)
# gbm.fit(train,y)
predictions = gbm.predict(X_dev)

p, r, f, _ = precision_recall_fscore_support(y_dev, predictions, pos_label=1, average="binary")

print('Precision:{}  Recall:{}  F1:{}'.format(p,r,f))

print('Accuracy: {}'.format(accuracy_score(y_dev, predictions)))

X_test = pd.read_csv('./embs/X_test.csv').iloc[:,1:]
test_stat = pd.read_csv('./user_rumor/test_scaled_stat_feat_df.csv')
test_stat.drop(columns=train_zero, inplace=True)
test_stat.drop(columns=['tweet_id'], inplace=True)
X_test = pd.concat([X_test.iloc[:,:768],test_stat],axis=1)

predictions = gbm.predict(X_test)
pred_dict = {'Id':[i for i in range(len(predictions))], 'Predicted':predictions}
pred_df = DataFrame(pred_dict)

pred_df.Predicted = pred_df.Predicted.apply(lambda x: 1 if x >= 0.5 else 0)

pred_df.to_csv('predictions.csv', index=False)
