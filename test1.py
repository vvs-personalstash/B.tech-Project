import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_auc_score, precision_recall_curve

from src.utils import evaluate_logits_all
def evaluate_methods(df):
    methods = df.columns[3:]    # Get the column names of evaluation methods
    results = pd.DataFrame(index=methods,
                           columns=["Sensitivity", "Specificity", "PPV", "NPV", "F1-score", "AUROC", "AUPRC","Accuracy"])
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


fpr, tpr, thresh = precision_recall_curve(y_true, y_pred, pos_label=1)
fpr1, tpr1, thresh1 = precision_recall_curve(y_true, y_pred2,   pos_label=1)
fpr2, tpr2, thresh2 = precision_recall_curve(y_true, y_pred_orig, pos_label=1)
fpr3, tpr3, thresh3 = precision_recall_curve(y_true, y_pred_orig_wossn, pos_label=1)
fpr4, tpr4, thresh4 = precision_recall_curve(y_true, y_pred_orig_woct,pos_label=1)
fpr5, tpr5, thresh5 = precision_recall_curve(y_true, y_pred_orig_ubibrowser_wo_domain_motif,pos_label=1)
fpr6, tpr6, thresh6 = precision_recall_curve(y_true, y_pred_orig_rf,pos_label=1)
fpr7, tpr7, thresh7 = precision_recall_curve(y_true, y_pred_orig_xg,pos_label=1)
fpr8, tpr8, thresh8 = precision_recall_curve(y_true, y_pred_orig_ubi,pos_label=1)
fpr9, tpr9, thresh9 = precision_recall_curve(y_true, y_pred_orig_lr,pos_label=1)
fpr10, tpr10, thresh10 = precision_recall_curve(y_true, y_pred_orig_knn,pos_label=1)
#roc_auc = roc_auc_score(fpr, tpr)
# Plot the ROC curve
plt.figure()  
plt.plot(tpr,fpr, label='Our Model: %0.3f)'%p['AUPRC'])# % roc_auc)
#plt.plot(fpr1,tpr1, label='UB2_Browser_1: (area = %0.2f)'%p['AUROC'])# % roc_auc)
plt.plot(tpr2,fpr2, label='TransDSI: (area = %0.3f)'%results_df.iloc[2,6])
plt.plot(tpr3,fpr3, label='Without SSN: (area = %0.3f)'%results_df.iloc[3,6])
plt.plot(tpr4,fpr4, label='Without CT: (area = %0.3f)'%results_df.iloc[4,6])
plt.plot(tpr5,fpr5, label='Ubibrowser_wo_domain: (area = %0.3f)'%results_df.iloc[1,6])
plt.plot(tpr6,fpr6, label='RF: (area = %0.3f)'%results_df.iloc[5,6])
plt.plot(tpr7,fpr7, label='XG: (area = %0.3f)'%results_df.iloc[6,6])
plt.plot(tpr8,fpr8, label='UBI: (area = %0.3f)'%results_df.iloc[0,6])
plt.plot(tpr9,fpr9, label='LR: (area = %0.3f)'%results_df.iloc[9,6])
plt.plot(tpr10,fpr10, label='KNN: (area = %0.3f)'%results_df.iloc[8,6])
#plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('AUPRC Curve(5-fold Cross Validation)')
plt.legend()
plt.show()