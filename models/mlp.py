import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["OMP_NUM_THREADS"] = "16"
os.environ["MKL_NUM_THREADS"] = "16"



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

import numpy as np
from stack_features import load_data

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

manual_size = 18

# manual = True
# name = "manual_only"
# size = 0

def run(size, manual, name):
    X_train, y_train, X_test, y_test = load_data(name)

    # Convert the labels to numpy arrays
    y_train = y_train.numpy() if isinstance(y_train, torch.Tensor) else np.array(y_train)
    y_test = y_test.numpy() if isinstance(y_test, torch.Tensor) else np.array(y_test)


    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)  # Add a dimension for the target
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)  # Add a dimension for the target

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            


    # MLP




    class MLP(nn.Module):
        def __init__(self, input_size, hidden_size, num_classes):
            super(MLP, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size)
            self.fc2 = nn.Linear(hidden_size, hidden_size)
            self.fc3 = nn.Linear(hidden_size, num_classes)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    # Hyperparameters
    input_size = manual_size*(1 if manual else 0) + 768*size  # Number of features in the dataset
    hidden_size = 128  # Number of neurons in hidden layers
    num_classes = 2  # Binary classification (adjust if more classes)

    # Instantiate the model
    model = MLP(input_size, hidden_size, num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Convert to torch tensors if not already
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32) if not torch.is_tensor(X_train) else X_train
    y_train_tensor = torch.tensor(y_train, dtype=torch.long) if not torch.is_tensor(y_train) else y_train
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32) if not torch.is_tensor(X_test) else X_test
    y_test_tensor = torch.tensor(y_test, dtype=torch.long) if not torch.is_tensor(y_test) else y_test

    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    def train_model(model, criterion, optimizer, train_loader, num_epochs=20):
        model.train()
        for epoch in range(num_epochs):
            for inputs, targets in train_loader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            if (epoch+1) % 5 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


    mlp_result = []

    for i in range(3):

        train_model(model, criterion, optimizer, train_loader, num_epochs=20)
        model.eval()

        with torch.no_grad():
            outputs = model(X_test_tensor)
            _, predicted = torch.max(outputs, 1)

        # Convert to numpy arrays
        y_test_np = y_test_tensor.numpy()
        predicted_np = predicted.numpy()
        probabilities_np = torch.softmax(outputs, dim=1)[:, 1].numpy()

        # Calculate metrics
        accuracy = np.mean(predicted_np == y_test_np)
        roc_auc = roc_auc_score(y_test_np, probabilities_np)
        precision = precision_score(y_test_np, predicted_np)
        recall = recall_score(y_test_np, predicted_np)
        f1 = f1_score(y_test_np, predicted_np)
        prauc = precision_score(y_test_np, predicted_np)


        print("MLP:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"PR AUC: {prauc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        mlp_result.append([accuracy, roc_auc, prauc, precision, recall, f1])
        
    avg_accuracy, std_accuracy = np.mean([x[0] for x in mlp_result]), np.std([x[0] for x in mlp_result])
    avg_rocauc, std_rocauc = np.mean([x[1] for x in mlp_result]), np.std([x[1] for x in mlp_result])
    avg_prauc, std_prauc = np.mean([x[2] for x in mlp_result]), np.std([x[2] for x in mlp_result])
    avg_precision , std_precision = np.mean([x[3] for x in mlp_result]), np.std([x[3] for x in mlp_result])
    avg_recall , std_recall = np.mean([x[4] for x in mlp_result]), np.std([x[4] for x in mlp_result])
    avg_f1 , std_f1 = np.mean([x[5] for x in mlp_result]), np.std([x[5] for x in mlp_result])



    with open(f'results/mlp_{name}.txt', 'w') as file:

        file.write(f"MLP Results {name}:\n")
        file.write(f"Accuracy: {avg_accuracy:.4f} ± {std_accuracy:.4f}\n")
        file.write(f"ROC AUC: {avg_rocauc:.4f} ± {std_rocauc:.4f}\n")
        file.write(f"PR AUC: {avg_prauc:.4f} ± {std_prauc:.4f}\n")
        file.write(f"Precision: {avg_precision:.4f} ± {std_precision:.4f}\n")
        file.write(f"Recall: {avg_recall:.4f} ± {std_recall:.4f}\n")
        file.write(f"F1 Score: {avg_f1:.4f} ± {std_f1:.4f}\n")
    
if __name__ == "__main__":
    
    # run(0, True, "manual_only")
    
    # run(2, False, "bert_only")
    # run(2, False, "llm_only")
    # run(4, False, "bert_llm")
    
    # run(2, False, "criteria_only")
    # run(4, False, "origin")
    
    # run(6, False, 'except_onehot')
    # run(6, True, 'all_features')
    
    run(4, True, "manual_origin")
    
