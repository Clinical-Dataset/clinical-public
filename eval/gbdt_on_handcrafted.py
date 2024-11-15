import pandas as pd
import json
import os
from tqdm import tqdm
import numpy as np
import lightgbm as lgb
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from lightgbm import LGBMClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_selector as selector

current_file_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from __init__ import partition_criteria

trial_df = pd.read_csv(f'{current_file_path}/data/trial_data.csv', sep='\t')



def train(training_set):

    
    X = training_set
    y = trial_df['label']
    
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)
    
    oversampler = RandomOverSampler()
    X_train_resampled, y_train_resampled = oversampler.fit_resample(X_encoded, y)
    
    X_train, X_test, y_train, y_test = train_test_split(X_train_resampled, y_train_resampled, test_size=0.2)


    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
    }

    gbm = lgb.train(
        params,
        dtrain,
        num_boost_round=100,
        valid_sets=[dtrain, dtest],
        valid_names=['train', 'test'],
        callbacks=[lgb.early_stopping(10)]
    )

    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    y_pred_binary = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    auc_roc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred_binary)

    print(f'auc: {auc_roc}, accuracy: {accuracy}, f1: {f1}')

    return auc_roc, accuracy


if __name__ == "__main__":
    

    features = [
        trial_df[['gender']],
        trial_df[['phase']],
        trial_df[['drugs']],
        trial_df[['diseases']],
        trial_df[['gender', 'phase']],
        trial_df[['gender', 'drugs']],
        trial_df[['gender', 'diseases']],
        trial_df[['phase', 'drugs']],
        trial_df[['phase', 'diseases']],
        trial_df[['drugs', 'diseases']],
        trial_df[['gender', 'phase', 'drugs']],
        trial_df[['gender', 'phase', 'diseases']],
        trial_df[['gender', 'drugs', 'diseases']],
        trial_df[['phase', 'drugs', 'diseases']],
        trial_df[['gender', 'phase', 'drugs', 'diseases']]
    ]



    result = [
        ['gender'],
        ['phase'],
        ['drugs'],
        ['diseases'],
        ['gender', 'phase'],
        ['gender', 'drugs'],
        ['gender', 'diseases'],
        ['phase', 'drugs'],
        ['phase', 'diseases'],
        ['drugs', 'diseases'],
        ['gender', 'phase', 'drugs'],
        ['gender', 'phase', 'diseases'],
        ['gender', 'drugs', 'diseases'],
        ['phase', 'drugs', 'diseases'],
        ['gender', 'phase', 'drugs', 'diseases'],
    ]
    
    result = {
        'feature': 
            ['gender',
            'phase',
            'drugs',
            'diseases',
            'gender + phase',
            'gender + drugs',
            'gender + diseases',
            'phase + drugs',
            'phase + diseases',
            'drugs + diseases',
            'gender + phase + drugs',
            'gender + phase + diseases',
            'gender + drugs + diseases',
            'phase + drugs + diseases',
            'gender + phase + drugs + diseases'],
        'auc_roc': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None],
        'accuracy': [None, None, None, None, None, None, None, None, None, None, None, None, None, None, None]      
    }

    num_iteration = 5
    
    for i in range(len(features)):
        
        total_auc_roc, total_accuracy = 0, 0
        
        for j in range (num_iteration):
            auc_roc, accuracy = train(features[i])
            total_auc_roc += auc_roc
            total_accuracy += accuracy
        
        avg_auc_roc, avg_accuracy = total_auc_roc / num_iteration, total_accuracy / num_iteration
        
        result["auc_roc"][i] = avg_auc_roc
        result["accuracy"][i] = avg_accuracy
        
    print(result)
    
    df = pd.DataFrame(result)

    # Save DataFrame to a CSV file
    
    df.to_csv('results/str_features.csv', index=False)