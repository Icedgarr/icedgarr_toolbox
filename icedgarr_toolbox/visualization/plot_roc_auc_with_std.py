import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc


def plot_roc_auc_with_std(data, model_name, target_column, prediction_column, split_column, color):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)
    for week in data[split_column].unique():
        fpr, tpr, _ = roc_curve(data.loc[data[split_column] == week, target_column],
                        data.loc[data[split_column] == week, prediction_column])
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    confidence_interval_auc = 2*np.std(aucs)/5
    plt.plot(mean_fpr, mean_tpr, color=color,
            label=r'%s (AUROC = %0.3f $\pm$ %0.3f)' % (model_name, mean_auc, confidence_interval_auc),
            lw=2, alpha=.8)

    confidence_interval_tpr = 2*np.std(tprs, axis=0)/5
    tprs_upper = np.minimum(mean_tpr + confidence_interval_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - confidence_interval_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=color, alpha=.2)



"""
Example of usage:
lt.figure(figsize=(10,10))#figsize=(10,10)
plot_roc_auc_with_std(test_data, 'XGBoost general', color=(0.00392156862745098, 0.45098039215686275, 0.6980392156862745))
plot_roc_auc_with_std(test_data_xgboost_diagnosis, 'XGBoost per diagnosis', color=(0.8705882352941177, 0.5607843137254902, 0.0196078431372549))
plot_roc_auc_with_std(test_data_baseline_clinical, 'Clinical baseline', color=(0.00784313725490196, 0.6196078431372549, 0.45098039215686275))
plot_roc_auc_with_std(test_data_baseline_diagnosis, 'Diagnosis baseline', color=(0.8352941176470589, 0.3686274509803922, 0.0))
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='black',
            label='Random', alpha=.8)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.legend(loc="lower right")
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.savefig('nature_medicine_paper/final_submission/auc_plot.pdf', format='pdf', dpi=400)
plt.show()
"""