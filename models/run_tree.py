#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

from tqdm import tqdm
import pandas as pd
import numpy as np
import lightgbm as lgb

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, average_precision_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
import os
import ast 


# In[2]:


import sys
sys.path.append('../')

from preprocess.protocol_encode import protocol2feature, load_sentence_2_vec, get_sentence_embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# In[3]:


sentence2vec = load_sentence_2_vec("../data") 


# In[4]:


train_data = pd.read_csv(f'../data/reason_prediction_train.csv', sep='\t')
test_data = pd.read_csv(f'../data/reason_prediction_test.csv', sep='\t')

train_data, valid_data = train_test_split(train_data, test_size=0.2, random_state=0)
print(train_data.head())


# In[5]:


# Missing Value Handling
train_data['criteria'].fillna('', inplace=True)
valid_data['criteria'].fillna('', inplace=True)
test_data['criteria'].fillna('', inplace=True)


# In[6]:


# # 32 sentences length can cover 95% of the data 

# criteria_lst = train_data['criteria']

# in_criteria_lengths = []
# ex_criteria_lengths = []

# for criteria in criteria_lst:
#     in_criteria, ex_criteria = protocol2feature(criteria, sentence2vec)
#     in_criteria_lengths.append(len(in_criteria))
#     ex_criteria_lengths.append(len(ex_criteria))

# print(f"Inclusion: {pd.Series(in_criteria_lengths).describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.999])}")
# print(f"Exclusion: {pd.Series(ex_criteria_lengths).describe(percentiles=[0.5, 0.75, 0.9, 0.95, 0.99, 0.999])}")


# In[7]:


def criteria2embedding(criteria_lst):
    criteria_lst = [protocol2feature(criteria, sentence2vec) for criteria in criteria_lst]

    incl_criteria = []
    excl_criteria = []

    for criteria in criteria_lst:
        incl_criteria.append(torch.mean(criteria[0], dim=0))
        excl_criteria.append(torch.mean(criteria[1], dim=0))

    incl_emb = torch.stack(incl_criteria)
    excl_emb = torch.stack(excl_criteria)

    return torch.cat((incl_emb, excl_emb), dim=1)


# In[8]:


criteria_train = criteria2embedding(train_data['criteria'])
criteria_valid = criteria2embedding(valid_data['criteria'])
criteria_test = criteria2embedding(test_data['criteria'])

LE = LabelEncoder()
LE.fit(train_data['reason'])

train_data['reason_encoded'] = LE.transform(train_data['reason'])
valid_data['reason_encoded'] = LE.transform(valid_data['reason'])
test_data['reason_encoded'] = LE.transform(test_data['reason'])


y_train = LE.transform(train_data['reason'])
y_valid = LE.transform(valid_data['reason'])
y_test = LE.transform(test_data['reason'])


# In[9]:


from transformers import AutoTokenizer, AutoModel
model = AutoModel.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")


# In[10]:


# # drug name to embedding
# load biobert model




def drug2embedding(drug_lst):
    
    # local_model_path = os.path.expanduser('~/FailureReasoning/trial-failure-reason-prediction/biobert-base-cased-v1.2')
    
    # local_model_path = os.path.abspath(local_model_path)
    # tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    # model = AutoModel.from_pretrained(local_model_path)
    # model_name = "dmis-lab/biobert-base-cased-v1.2"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)
     
    drug_emb = []
    for drugs in tqdm(drug_lst):
        if len(drugs) == 0:
            print("Warning: Empty drug list is found")
            drug_emb.append(torch.zeros(768, dtype=torch.float32))
        else:
            # mean pooling
            drugs_emb = torch.mean(torch.stack([get_sentence_embedding(drug, tokenizer, model) for drug in drugs.split(';')]), dim=0)
            drug_emb.append(drugs_emb)
    
    return torch.stack(drug_emb)


# In[11]:


# # Disease name to embedding
def disease2embedding(disease_lst):
    
    # local_model_path = os.path.expanduser('~/FailureReasoning/trial-failure-reason-prediction/biobert-base-cased-v1.2')
    # local_model_path = os.path.abspath(local_model_path)
    # tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    # model = AutoModel.from_pretrained(local_model_path)
    # model_name = "dmis-lab/biobert-base-cased-v1.2"
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)
     
    disease_emb = []
    for diseases in tqdm(disease_lst):
        if len(diseases) == 0:
            print("Warning: Empty disease list is found")
            disease_emb.append(torch.zeros(768, dtype=torch.float32))
        else:
            # mean pooling
            diseases_emb = torch.mean(torch.stack([get_sentence_embedding(disease, tokenizer, model) for disease in diseases.split(';')]), dim=0)
            disease_emb.append(diseases_emb)
    
    return torch.stack(disease_emb)



def icds2embedding(nctid_lst):
    icds_emb = []
    raw_data = pd.read_csv('../data/raw_data.csv')
    icd_emb_data = pd.read_csv('../data/icd-10-cm-2022-0050.csv')
    embedding_cols = [f'V{i}' for i in range(1, 51)]
    
    for nctid in tqdm(nctid_lst, desc='ICD Embedding'):
        #print(f"nctid {nctid} is found")
        
        # Extract the ICD code list string and safely evaluate it
        icd_codes_str = raw_data[raw_data['nctid'] == nctid]['icdcodes'].values
        
        if len(icd_codes_str) == 0:
            #print(f"Warning: Empty icd list is found for nctid {nctid}")
            icds_emb.append(torch.zeros(50, dtype=torch.float32))
            print(f"No icd code for nctid {nctid}")
            continue

        # Assuming each nctid has one entry, hence taking the first element
        icd_codes_list = ast.literal_eval(icd_codes_str[0])
        
        # Flatten the list of lists
        icd_codes = [code for sublist in icd_codes_list for code in ast.literal_eval(sublist)]
        
        if len(icd_codes) == 0:
            icds_emb.append(torch.zeros(50, dtype=torch.float32))
            print(f"No icd code for nctid {nctid}")
        else:
            # Mean pooling
            icd_emb_list = []
            for icd_code in icd_codes:
                try:

                    #print(f"icd code {icd_code} is found")
                    # Parse and clean icd code
                    cleaned_idc_code = ''.join(e for e in icd_code if e.isalnum()).upper()
                    # Find the embedding for the cleaned ICD code
                    icd_row = icd_emb_data[icd_emb_data['code'] == cleaned_idc_code]
                    embedding = icd_row.iloc[0][embedding_cols].values
                    #print(f"embedding {embedding} is found")
                    embedding = embedding.astype(float)
                    icd_emb_list.append(torch.tensor(embedding, dtype=torch.float32))
                    #print(f"icd emb list appended")
                except Exception as e:
                    # print(f"Warning: ICD code {icd_code} not found in the embedding data")
                    # print(f"Error: {e}")
                    continue
            
            if icd_emb_list:
                # Mean pooling: average the collected embeddings
                mean_embedding = torch.mean(torch.stack(icd_emb_list), dim=0)
                icds_emb.append(mean_embedding)
            else:
                # If no valid ICD codes were found, append a zero tensor
                icds_emb.append(torch.zeros(50, dtype=torch.float32))
    
    return icds_emb
    






def smiles2embedding(nctid_lst):


    smiless_emb = []
    raw_data = pd.read_csv('../data/raw_data.csv')  # Load the raw data

    for nctid in tqdm(nctid_lst, desc='SMILES Embedding'):
        # Extract the SMILES list for the current nctid
        smiles_str = raw_data[raw_data['nctid'] == nctid]['smiless'].values
        
        if len(smiles_str) == 0:
            # No SMILES strings found for this nctid
            smiless_emb.append(torch.zeros(768, dtype=torch.float32))
            print(f"No SMILES strings for nctid {nctid}")
            continue

        smiles_list = ast.literal_eval(smiles_str[0])  
        
        if len(smiles_list) == 0:
            # No valid SMILES strings
            smiless_emb.append(torch.zeros(768, dtype=torch.float32))
            print(f"No SMILES strings for nctid {nctid}")
        else:
            smiles_emb_list = []
            for smiles in smiles_list:
                try:
                    embedding = get_sentence_embedding(smiles, tokenizer, model)
                    smiles_emb_list.append(embedding)
                except Exception as e:
                    print(f"Error processing SMILES: {smiles} for nctid {nctid}")
                    continue
            
            if smiles_emb_list:
                # Mean pooling: average the collected embeddings
                mean_embedding = torch.mean(torch.stack(smiles_emb_list), dim=0)
                smiless_emb.append(mean_embedding)
                # max pooling
                #max_embedding = torch.max(torch.stack(smiles_emb_list), dim=0)
                #smiless_emb.append(max_embedding)
            else:
                smiless_emb.append(torch.zeros(768, dtype=torch.float32))
    
    return smiless_emb

    



if not os.path.exists('../data/drug_emb.pt') or not os.path.exists('../data/disease_emb.pt'):
    drug_emb = {}
    drug_emb['train'] = drug2embedding(train_data['drugs'].tolist())
    drug_emb['valid'] = drug2embedding(valid_data['drugs'].tolist())
    drug_emb['test'] = drug2embedding(test_data['drugs'].tolist())

    disease_emb = {}
    disease_emb['train'] = disease2embedding(train_data['diseases'].tolist())
    disease_emb['valid'] = disease2embedding(valid_data['diseases'].tolist())
    disease_emb['test'] = disease2embedding(test_data['diseases'].tolist())

    torch.save(drug_emb, '../data/drug_emb.pt')
    torch.save(disease_emb, '../data/disease_emb.pt')
else:
    drug_emb = torch.load('../data/drug_emb.pt')
    disease_emb = torch.load('../data/disease_emb.pt')
    
    
if not os.path.exists('../data/icd_emb.pt'):
    icd_emb = {}
    icd_emb['train'] = icds2embedding(train_data['nctid'].tolist())
    icd_emb['valid'] = icds2embedding(valid_data['nctid'].tolist())
    icd_emb['test'] = icds2embedding(test_data['nctid'].tolist())
    torch.save(icd_emb, '../data/icd_emb.pt')
else:
    icd_emb = torch.load('../data/icd_emb.pt')
    




# nctid_lst = train_data['nctid'].tolist()
# smiles2embedding(nctid_lst)

if not os.path.exists('../data/smiles_emb.pt'):
    smiles_emb = {}
    smiles_emb['train'] = smiles2embedding(train_data['nctid'].tolist())
    smiles_emb['valid'] = smiles2embedding(valid_data['nctid'].tolist())
    smiles_emb['test'] = smiles2embedding(test_data['nctid'].tolist())
    torch.save(smiles_emb, '../data/smiles_emb.pt')

else:
    smiles_emb = torch.load('../data/smiles_emb.pt')
    



# breakpoint()

# In[13]:


encoder = OneHotEncoder(sparse_output=False)
encoder.fit(train_data[['phase']])

phase_emb = {}
phase_emb['train'] = torch.tensor(encoder.transform(train_data[['phase']])).float()
phase_emb['valid'] = torch.tensor(encoder.transform(valid_data[['phase']])).float()
phase_emb['test'] = torch.tensor(encoder.transform(test_data[['phase']])).float()



# print shape of icd_emb
print(f"Shape of icd_emb: {icd_emb['train'][0].shape}")
# print length of icd_emb
print(f"Length of icd_emb train: {len(icd_emb['train'])}")
print(f"Length of icd_emb valid: {len(icd_emb['valid'])}")
print(f"Length of icd_emb test: {len(icd_emb['test'])}")


# print shape of drug_emb
print(f"Shape of drug_emb: {drug_emb['train'][0].shape}")
# print length of drug_emb
print(f"Length of drug_emb train: {len(drug_emb['train'])}")
print(f"Length of drug_emb valid: {len(drug_emb['valid'])}")
print(f"Length of drug_emb test: {len(drug_emb['test'])}")

print(f"Shape of smiles_emb: {smiles_emb['train'][0].shape}")
# print length of smiles_emb
print(f"Length of smiles_emb train: {len(smiles_emb['train'])}")
print(f"Length of smiles_emb valid: {len(smiles_emb['valid'])}")
print(f"Length of smiles_emb test: {len(smiles_emb['test'])}")

icd_emb['train'] = torch.stack(icd_emb['train'])
icd_emb['valid'] = torch.stack(icd_emb['valid'])
icd_emb['test'] = torch.stack(icd_emb['test'])

smiles_emb['train'] = torch.stack(smiles_emb['train'])
smiles_emb['valid'] = torch.stack(smiles_emb['valid'])
smiles_emb['test'] = torch.stack(smiles_emb['test'])

print(f"Type of icd_emb['train']: {type(icd_emb['train'])}")
print(f"Type of icd_emb['train'][0]: {type(icd_emb['train'][0])}")


# In[14]:

# X_train = torch.cat([criteria_train, drug_emb['train'], disease_emb['train'], phase_emb['train']], 1)
# X_valid = torch.cat([criteria_valid, drug_emb['valid'], disease_emb['valid'], phase_emb['valid']], 1)
# X_test = torch.cat([criteria_test, drug_emb['test'], disease_emb['test'], phase_emb['test']], 1)

X_train = torch.cat([criteria_train, drug_emb['train'], disease_emb['train'], phase_emb['train'], icd_emb['train'], smiles_emb['train']], 1)
X_valid = torch.cat([criteria_valid, drug_emb['valid'], disease_emb['valid'], phase_emb['valid'], icd_emb['valid'], smiles_emb['valid']], 1)
X_test = torch.cat([criteria_test, drug_emb['test'], disease_emb['test'], phase_emb['test'], icd_emb['test'], smiles_emb['test']], 1)


# In[15]:


def evaluate_model(y_test, y_pred, y_pred_proba, model_name):
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_scores = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average=None)
    f1_scores = f1_score(y_test, y_pred, average=None)
    precision_scores = precision_score(y_test, y_pred, average=None)
    recall_scores = recall_score(y_test, y_pred, average=None)
    pr_auc_scores = average_precision_score(y_test, y_pred_proba, average=None)

    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'ROC-AUC': auc_scores.mean(),
        'F1 Score': f1_scores.mean(),
        'Precision': precision_scores.mean(),
        'Recall': recall_scores.mean(),
        'PR-AUC': pr_auc_scores.mean()
    }


# In[16]:


def shuffle_and_split(X_train, X_valid, X_test, y_train, y_valid, y_test):
    # Concatenate all data and labels
    X_all = torch.cat((X_train, X_valid, X_test), dim=0)
    y_all = np.concatenate((y_train, y_valid, y_test), axis=0)

    # Convert y_all to tensor
    y_all_tensor = torch.tensor(y_all)

    # Create a TensorDataset and DataLoader for shuffling
    dataset = TensorDataset(X_all, y_all_tensor)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=True)

    # Get shuffled data
    X_all_shuffled, y_all_shuffled = next(iter(loader))

    # Convert y_all_shuffled back to numpy array
    y_all_shuffled = y_all_shuffled.numpy()

    # Get original sizes
    train_size = len(X_train)
    valid_size = len(X_valid)
    test_size = len(X_test)

    # Split shuffled data into new train, valid, and test sets
    X_train_new = X_all_shuffled[:train_size]
    y_train_new = y_all_shuffled[:train_size]
    X_valid_new = X_all_shuffled[train_size:train_size+valid_size]
    y_valid_new = y_all_shuffled[train_size:train_size+valid_size]
    X_test_new = X_all_shuffled[train_size+valid_size:]
    y_test_new = y_all_shuffled[train_size+valid_size:]

    return X_train_new, X_valid_new, X_test_new, y_train_new, y_valid_new, y_test_new


# In[17]:


# # Models to evaluate
# models = {
#     'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0, n_jobs=4),
#     'XGBoost': xgb.XGBClassifier(objective='multi:softprob', num_class=6, learning_rate=0.01, max_depth=10, n_estimators=100, verbosity=1, n_jobs=4),
#     'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs', n_jobs=4),
#     'AdaBoost': AdaBoostClassifier(n_estimators=50, learning_rate=0.5)
# }


    
# Evaluate the LightGBM model
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_valid, y_valid, reference=lgb_train)
lgb_params = {
    'boosting': 'gbdt',
    'objective': 'multiclass',
    'metric': 'multi_logloss',
    'num_class': 6,
    'learning_rate': 0.01,
    'early_stopping_round': 10,
    'verbosity': 1,
    'max_depth': 10,
    'num_threads': 4
}
gbm = lgb.train(lgb_params, lgb_train, num_boost_round=500, valid_sets=[lgb_eval], callbacks=[lgb.log_evaluation()])
y_pred_lgb = gbm.predict(X_test, num_iteration=gbm.best_iteration)
metrics = evaluate_model(y_test, np.argmax(y_pred_lgb, axis=1), y_pred_lgb, 'Gradient Boost Decision Tree')

# display metrics
print(metrics)
    

