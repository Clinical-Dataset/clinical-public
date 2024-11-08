from statsmodels.stats.contingency_tables import mcnemar
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# read data
y_pred_lr = np.load('y_pred_lr.npy')
y_pred_lr = (y_pred_lr > 0.5).astype(int)
y_pred_dcn = np.load('y_pred_dcn.npy')
y_pred_dcn = (y_pred_dcn > 0.5).astype(int)

y_true = np.load('y_test_dcn.npy')

print(y_pred_lr[:20])
print(y_pred_dcn[:20])


# Initialize counts
n00, n01, n10, n11 = 0, 0, 0, 0

# Calculate the counts
for lr_pred, dcn_pred, true in zip(y_pred_lr, y_pred_dcn, y_true):
    if lr_pred == true and dcn_pred == true:
        n00 += 1
    elif lr_pred == true and dcn_pred != true:
        n01 += 1
    elif lr_pred != true and dcn_pred == true:
        n10 += 1
    elif lr_pred != true and dcn_pred != true:
        n11 += 1

# Create the contingency table
contingency_table = np.array([[n00, n01],
                              [n10, n11]])

# Perform McNemar's test
result = mcnemar(contingency_table, exact=True)

print(f'statistic={result.statistic}, p-value={result.pvalue}')

# Interpret the result
alpha = 0.05
if result.pvalue < alpha:
    print('Significant difference (reject H0)')
else:
    print('No significant difference (fail to reject H0)')




