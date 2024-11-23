import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

from src.utils import evaluate_logits_all
def evaluate_methods(df):
    methods = df.columns[3:]    # Get the column names of evaluation methods
    results = pd.DataFrame(index=methods,
                           columns=["Sensitivity", "Specificity", "PPV", "NPV", "F1-score", "AUROC", "AUPRC"])
    Y_val = df["label"]

    for method in methods:
        Y_pred_prob = df[method]
        perf = evaluate_logits_all(Y_val, Y_pred_prob)
        results.loc[method] = perf

    return results


# Load the data
ub2 = pd.read_csv("/Users/vv/Desktop/btp_project/TransDSI/results/performance/GSD/UB2_TransDSI_crossval1.csv", sep=",")
prev_results=pd.read_csv("/Users/vv/Desktop/btp_project/TransDSI/results/performance/GSD/GSD_crossval_prob.csv")
y_true = ub2.iloc[:, 2].astype(np.float32)
y_pred = ub2.iloc[:, 5].astype(np.float32)
y_pred2= ub2.iloc[:, 3].astype(np.float32)
p=evaluate_logits_all(y_true,y_pred)
print(p)
results_df = evaluate_methods(prev_results)
print(results_df)
y_pred_orig=prev_results.iloc[:,5].astype(np.float32)
y_pred_orig_wossn=prev_results.iloc[:,6].astype(np.float32)
y_pred_orig_woct=prev_results.iloc[:,7].astype(np.float32)
y_pred_orig_ubibrowser_wo_domain_motif=prev_results.iloc[:,4].astype(np.float32)
y_pred_orig_rf=prev_results.iloc[:,8].astype(np.float32)
y_pred_orig_xg=prev_results.iloc[:,9].astype(np.float32)
y_pred_orig_knn=prev_results.iloc[:,11].astype(np.float32)
y_pred_orig_lr=prev_results.iloc[:,12].astype(np.float32)
y_pred_orig_ubi=prev_results.iloc[:,4].astype(np.float32)
#y_pred_orig=prev_results.iloc[:,5].astype(np.float32)
q=evaluate_logits_all(y_true,y_pred_orig)

# Assuming the true labels are in the third column and predictions are in the sixth column


fpr, tpr, thresh = roc_curve(y_true, y_pred)
fpr1, tpr1, thresh1 = roc_curve(y_true, y_pred2)
fpr2, tpr2, thresh2 = roc_curve(y_true, y_pred_orig)
fpr3, tpr3, thresh3 = roc_curve(y_true, y_pred_orig_wossn)
fpr4, tpr4, thresh4 = roc_curve(y_true, y_pred_orig_woct)
fpr5, tpr5, thresh5 = roc_curve(y_true, y_pred_orig_ubibrowser_wo_domain_motif)
fpr6, tpr6, thresh6 = roc_curve(y_true, y_pred_orig_rf)
fpr7, tpr7, thresh7 = roc_curve(y_true, y_pred_orig_xg)
fpr8, tpr8, thresh8 = roc_curve(y_true, y_pred_orig_ubi)
fpr9, tpr9, thresh9 = roc_curve(y_true, y_pred_orig_lr)
fpr10, tpr10, thresh10 = roc_curve(y_true, y_pred_orig_knn)
#roc_auc = roc_auc_score(fpr, tpr)
# Plot the ROC curve
plt.figure()  
plt.plot(fpr,tpr, label='Our Model: %0.3f)'%p['AUROC'])# % roc_auc)
#plt.plot(fpr1,tpr1, label='UB2_Browser_1: (area = %0.2f)'%p['AUROC'])# % roc_auc)
plt.plot(fpr2,tpr2, label='TransDSI: (area = %0.3f)'%results_df.iloc[2,5])
plt.plot(fpr3,tpr3, label='Without SSN: (area = %0.3f)'%results_df.iloc[3,5])
plt.plot(fpr4,tpr4, label='Without CT: (area = %0.3f)'%results_df.iloc[4,5])
plt.plot(fpr5,tpr5, label='Ubibrowser_wo_domain: (area = %0.3f)'%results_df.iloc[1,5])
plt.plot(fpr6,tpr6, label='RF: (area = %0.3f)'%results_df.iloc[5,5])
plt.plot(fpr7,tpr7, label='XG: (area = %0.3f)'%results_df.iloc[6,5])
plt.plot(fpr8,tpr8, label='UBI: (area = %0.3f)'%results_df.iloc[0,5])
plt.plot(fpr9,tpr9, label='LR: (area = %0.3f)'%results_df.iloc[9,5])
plt.plot(fpr10,tpr10, label='KNN: (area = %0.3f)'%results_df.iloc[8,5])
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('1-Specificity')
plt.ylabel('Senstivity')
plt.title('ROC Curve(5-fold Cross Validation)')
plt.legend()
plt.show()
# random_probs = [0 for i in range(len(y_test))]
# p_fpr, p_tpr, _ = roc_curve(y_test, random_probs, pos_label=1)
# plt.plot(fpr1, tpr1, linestyle='--',color='orange', label='Logistic Regression')
# plt.plot(fpr2, tpr2, linestyle='--',color='green', label='KNN')
# plt.plot(p_fpr, p_tpr, linestyle='--', color='blue')
# #title
# plt.title('ROC curve')
# #x-label
# plt.xlabel('False Positive Rate')
# #y-label
# plt.ylabel('True Positive rate')
 
# plt.legend(loc='best')
# plt.savefig('ROC',dpi=300)
# plt.show();
# # Calculate sensitivity and specificity
# # sensitivity, specificity = calculate_sensitivity_specificity(y_true, y_pred)

# # # Plotting
# # plt.figure(figsize=(8, 6))
# # plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
# # plt.plot(specificity, sensitivity, 'ro', label='Sensitivity vs Specificity')
# # plt.xlabel('Specificity')
# # plt.ylabel('Sensitivity')
# # plt.title('Sensitivity vs Specificity')
# # plt.legend()
# # plt.grid(True)
# # plt.show()