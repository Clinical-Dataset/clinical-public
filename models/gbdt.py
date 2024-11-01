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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import average_precision_score



current_file_path = os.path.dirname(os.path.realpath(__file__))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



from stack_features import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, precision_recall_curve, auc



def gbm_all(name):

    X_train, y_train, X_test, y_test = load_data(name)

    # oversampler = RandomOverSampler()
    # X_train, y_train = oversampler.fit_resample(X_train, y_train)
    # X_test, y_test = oversampler.fit_resample(X_test, y_test)



    dtrain = lgb.Dataset(X_train, label=y_train)
    dtest = lgb.Dataset(X_test, label=y_test, reference=dtrain)
    
    params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'num_leaves': 127,
        'learning_rate': 0.005,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': 0,
    }

    gbm = lgb.train(
        params,
        dtrain,
        num_boost_round=50,
        valid_sets=[dtrain, dtest],
        valid_names=['train', 'test'],
        callbacks=[lgb.early_stopping(10)]
    )

    y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)

    y_pred_binary = (y_pred >= 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred_binary)
    roc_auc = roc_auc_score(y_test, y_pred_binary)
    #f1 = f1_score(y_test, y_pred)
    pr_auc = average_precision_score(y_test, y_pred)
    precision, recall, _ = precision_recall_curve(y_test, y_pred)
    precision = np.mean(precision)
    recall = np.mean(recall)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    print("Accuracy:", accuracy)
    print("ROC-AUC Score:", roc_auc)
    print("PR AUC:", pr_auc)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    

    return accuracy, roc_auc, precision, recall, pr_auc, f1





def test(name):
    results = []
    for _ in range(3):
        accuracy, roc_auc, precision, recall, pr_auc, f1 = gbm_all(name)
        results.append((accuracy, roc_auc, precision, recall, pr_auc, f1))

    avg_accuracy = np.mean([result[0] for result in results])
    avg_roc_auc = np.mean([result[1] for result in results])
    avg_precision = np.mean([result[2] for result in results])
    avg_recall = np.mean([result[3] for result in results])
    avg_pr_auc = np.mean([result[4] for result in results])
    avg_f1 = np.mean([result[5] for result in results])
    
    std_accuracy = np.std([result[0] for result in results])
    std_roc_auc = np.std([result[1] for result in results])
    std_precision = np.std([result[2] for result in results])
    std_recall = np.std([result[3] for result in results])
    std_pr_auc = np.std([result[4] for result in results])
    std_f1 = np.std([result[5] for result in results])

    print(f"Average Accuracy: {avg_accuracy}")
    print(f"Average ROC AUC: {avg_roc_auc}")
    print(f"Average Precision: {avg_precision}")
    print(f"Average Recall: {avg_recall}")
    print(f"Average PR AUC: {avg_pr_auc}")
    print(f"Average F1 Score: {avg_f1}")

    print(f"Standard Deviation Accuracy: {std_accuracy}")
    print(f"Standard Deviation ROC AUC: {std_roc_auc}")
    print(f"Standard Deviation Precision: {std_precision}")
    print(f"Standard Deviation Recall: {std_recall}")
    print(f"Standard Deviation PR AUC: {std_pr_auc}")
    print(f"Standard Deviation F1 Score: {std_f1}")

    # Save results to a file
    with open(f'results/gbdt_{name}.txt', 'w') as file:
        file.write(f"Results for gbdt_{name}\n")
        file.write(f"Average PR AUC: {avg_pr_auc}\n")
        file.write(f"Average F1 Score: {avg_f1}\n")
        file.write(f"Average ROC AUC: {avg_roc_auc}\n")
        file.write(f"Average Accuracy: {avg_accuracy}\n")
        file.write(f"Average Precision: {avg_precision}\n")
        file.write(f"Average Recall: {avg_recall}\n")
        
        file.write(f"Standard Deviation PR AUC: {std_pr_auc}\n")
        file.write(f"Standard Deviation F1 Score: {std_f1}\n")
        file.write(f"Standard Deviation ROC AUC: {std_roc_auc}\n")
        file.write(f"Standard Deviation Accuracy: {std_accuracy}\n")
        file.write(f"Standard Deviation Precision: {std_precision}\n")
        file.write(f"Standard Deviation Recall: {std_recall}\n")



if __name__ == "__main__":
    

    
    # test("manual_only")
    
    # test("origin")
    
    # test('bert_only')
    # test('llm_only')
    # test('bert_llm')
    

    # test('criteria_only')

    # test('all_features')
    
    # test('except_onehot')
    
    test("manual_origin")
    
   # test("onehot_only")


    
    

    
    
    
        
        