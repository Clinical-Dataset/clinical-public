import numpy as np
from sklearn.metrics import precision_recall_curve, auc
from sklearn.utils import resample

# Load predictions and true labels
y_pred_lr = np.load('y_pred_lr.npy')
y_pred_lr = (y_pred_lr > 0.5).astype(int)
y_pred_dcn = np.load('y_pred_dcn.npy')
y_pred_dcn = (y_pred_dcn > 0.5).astype(int)

y_true = np.load('y_test_dcn.npy')


import numpy as np
from sklearn.metrics import precision_recall_curve, auc

# Load predictions and true labels
y_pred_lr_prob = np.load('y_pred_lr.npy')
y_pred_dcn_prob = np.load('y_pred_dcn.npy')
y_true = np.load('y_test_dcn.npy')

# Compute PR-AUC for logistic regression
precision_lr, recall_lr, _ = precision_recall_curve(y_true, y_pred_lr_prob)
pr_auc_lr = auc(recall_lr, precision_lr)

# Compute PR-AUC for DCN
precision_dcn, recall_dcn, _ = precision_recall_curve(y_true, y_pred_dcn_prob)
pr_auc_dcn = auc(recall_dcn, precision_dcn)

print(f'PR-AUC Logistic Regression: {pr_auc_lr:.3f}')
print(f'PR-AUC DCN: {pr_auc_dcn:.3f}')


import numpy as np

def permutation_test(y_true, y_pred_prob1, y_pred_prob2, n_permutations=1000, random_seed=42):
    rng = np.random.RandomState(random_seed)
    pr_auc_diff = []
    
    # Original difference in PR-AUC
    precision1, recall1, _ = precision_recall_curve(y_true, y_pred_prob1)
    pr_auc1 = auc(recall1, precision1)
    
    precision2, recall2, _ = precision_recall_curve(y_true, y_pred_prob2)
    pr_auc2 = auc(recall2, precision2)
    
    original_diff = pr_auc2 - pr_auc1
    
    # Permutations
    for i in range(n_permutations):
        perm_indices = rng.permutation(len(y_true))
        y_pred_prob1_perm = y_pred_prob1[perm_indices]
        y_pred_prob2_perm = y_pred_prob2[perm_indices]
        
        precision1_perm, recall1_perm, _ = precision_recall_curve(y_true, y_pred_prob1_perm)
        pr_auc1_perm = auc(recall1_perm, precision1_perm)
        
        precision2_perm, recall2_perm, _ = precision_recall_curve(y_true, y_pred_prob2_perm)
        pr_auc2_perm = auc(recall2_perm, precision2_perm)
        
        pr_auc_diff.append(pr_auc2_perm - pr_auc1_perm)
    
    pr_auc_diff = np.array(pr_auc_diff)
    p_value = np.mean(pr_auc_diff >= original_diff)
    
    return original_diff, p_value

# Perform permutation test
original_diff, p_value = permutation_test(y_true, y_pred_lr_prob, y_pred_dcn_prob)

print(f'Original PR-AUC Difference: {original_diff:.3f}')
print(f'p-value: {p_value}')

# Interpretation
alpha = 0.05
if p_value < alpha:
    print('Significant difference (reject H0)')
else:
    print('No significant difference (fail to reject H0)')
