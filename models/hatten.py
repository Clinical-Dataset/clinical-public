import pandas as pd
import json
import os
from tqdm import tqdm
import numpy as np
from xml.etree import ElementTree as ET
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import sys
import os
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from stack_features import load_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss, precision_recall_curve, auc

os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"
torch.set_num_threads(16)



class CriteriaModel(nn.Module):
    def __init__(self, size):
        super(CriteriaModel, self).__init__()
        self.sentence_embedding_dim = 768

        self.transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.sentence_embedding_dim, nhead=2, dropout=0.2, batch_first=True, dim_feedforward=2*self.sentence_embedding_dim)
        layer_norm = nn.LayerNorm(self.sentence_embedding_dim)
        
        self.transformer_encoder = nn.TransformerEncoder(
            self.transformer_encoder_layer, num_layers=1, norm=layer_norm)
        
        self.fc1 = nn.Linear(size*self.sentence_embedding_dim, self.sentence_embedding_dim*size//2)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.sentence_embedding_dim*size//2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, size):
        x = x.reshape(-1, size, self.sentence_embedding_dim)

        x = self.transformer_encoder(x)
        x = x.reshape(-1, size*self.sentence_embedding_dim)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

def train_test_eval(params, size, name):
    
    lr = params['lr']
    batch_size = params['batch_size']
    optimizer = params['optimizer']
    weight_decay = params['weight_decay']
    

    X_train, y_train, X_test, y_test = load_data(name)
    
    # from imblearn.over_sampling import RandomOverSampler
    # oversampler = RandomOverSampler(random_state=0)
    # X_train, y_train = oversampler.fit_resample(X_train, y_train)
    # X_test, y_test = oversampler.fit_resample(X_test, y_test)
    
    # X_train, y_train  = torch.FloatTensor(X_train), torch.FloatTensor(y_train) 
    # X_test, y_test = torch.FloatTensor(X_test), torch.FloatTensor(y_test)
    
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    weight_for_positives = class_weights[1] 

    pos_weight = torch.tensor([weight_for_positives]).to(device)
    
    
    class CriteriaDataset(Dataset):
        def __init__(self, X, y):
            self.X = X
            self.y = y

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]

    train_dataset = CriteriaDataset(X_train, torch.tensor(y_train))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = CriteriaDataset(X_test, torch.tensor(y_test))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


    model = CriteriaModel(size).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight= pos_weight)  ##remove weight and do another test
    
    if optimizer == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    elif optimizer == 'Adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=1e-5)
    else:
        return False

    num_epochs = 50
    best_auc = 0
    for epoch in tqdm(range(num_epochs)):
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            y_pred = model(X_batch, size)
            loss = criterion(y_pred, y_batch.unsqueeze(1).float())
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test.to(device), size)
            y_pred_test = y_pred_test.cpu().numpy().flatten()

            y_true = y_test
            y_pred_probs = nn.Sigmoid()(torch.tensor(y_pred_test)).cpu().flatten()
            precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
            pr_auc = auc(recall, precision)

            #auc_test = roc_auc_score(y_test, y_pred_test)

            if pr_auc > best_auc:
                best_auc = pr_auc
                print(f"Epoch {epoch}\tBest PR_AUC: {pr_auc}, saving model...")

                torch.save(model.state_dict(), 'saves/enrollment_model.pt')
    # Final evaluation
    model.load_state_dict(torch.load('saves/enrollment_model.pt'))

    model.eval()
    with torch.no_grad():
        y_true = y_test
        y_pred_test = model(X_test.to(device), size)
        y_pred_probs = nn.Sigmoid()(y_pred_test).cpu().flatten()

        # Convert predicted probabilities to binary predictions
        threshold = 0.5
        y_pred = [1 if prob >= threshold else 0 for prob in y_pred_probs]
        accuracy = accuracy_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_pred_probs)
        f1 = f1_score(y_true, y_pred)
        # Calculate metrics
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print("ROC-AUC Score:", roc_auc_score(y_true, y_pred_probs))

        # Calculating PR AUC
        precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
        pr_auc = auc(recall, precision)
        precision = np.mean(precision)
        recall = np.mean(recall)
        print("PR AUC:", pr_auc)

        print("Precision:", precision_score(y_true, y_pred))
        print("Recall:", recall_score(y_true, y_pred))
        print("F1 Score:", f1_score(y_true, y_pred))
        print("Log Loss:", log_loss(y_true, y_pred_probs))
        
        return accuracy, roc_auc, precision, recall, pr_auc, f1



def test(size, name):
    
    
    params = {'batch_size': 256, 'lr': 0.0005, 'optimizer': 'Adagrad', 'weight_decay': 1e-5}
    
    results = []
    for _ in range(5):
        accuracy, roc_auc, precision, recall, pr_auc, f1 = train_test_eval(params, size, name)
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
    with open(f'results/hatten_{name}.txt', 'w') as file:
        file.write(f"Results for hatten_{name}\n")
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

    # params for test
    
    # test(2, 'bert_only')
    # test(2, 'llm_only')
    # test(4, 'bert_llm')
    
    #test(1, 'manual_only')
    # test(2, 'criteria_only')

    # test(7, 'all_features')
    
    # test(6, 'except_onehot')
    
    test(4, "origin")

    