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

def xgb_objective(trial):
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
    gamma = trial.suggest_float('gamma',1e-10,1e10,log=True)



    classifier_obj = xgb.XGBClassifier(objective = 'binary:logistic',
                                       use_label_encoder=False,
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
                                       gamma=gamma,
                                       random_state=42)

    classifier_obj.fit(X_train,y_train)
    predictions = classifier_obj.predict(X_dev).astype(int)
    if np.sum(predictions) == 0:
        predictions[0] = 1
    p, r, f, _ = precision_recall_fscore_support(y_dev, predictions, pos_label=1, average="binary")
    return f

study = optuna.create_study(direction='maximize')
study.optimize(xgb_objective, n_trials=500,n_jobs=-1)
