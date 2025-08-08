import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_pr_auc_with_std(data, model_name, target_column, prediction_column, split_column, color):
    precisions = []
    aps = []
    mean_recall = np.linspace(0, 1, 1000)
    for week in data[split_column].unique():
        precision, recall, _ = precision_recall_curve(data.loc[data[split_column] == week, target_column],
                                                      data.loc[data[split_column] == week, prediction_column])
        precision = precision[::-1]
        recall = recall[::-1]
        precisions.append(np.interp(mean_recall, recall, precision))
        aps.append(average_precision_score(data.loc[data[split_column] == week, target_column],
                                           data.loc[data[split_column] == week, prediction_column]))

    mean_precision = np.mean(precisions, axis=0)
    mean_ap = np.mean(aps)  # average_precision_score(data[target_column], data[prediction_column])
    confidence_interval_ap = 2 * np.std(aps) / 5
    plt.plot(mean_recall, mean_precision, color=color,
             label=r'%s (AP = %0.3f $\pm$ %0.3f)' % (model_name, mean_ap, confidence_interval_ap),
             lw=2, alpha=.8)

    confidence_interval_precisions = 2 * np.std(precisions, axis=0) / 5
    precision_upper = np.minimum(mean_precision + confidence_interval_precisions, 1)
    precision_lower = np.maximum(mean_precision - confidence_interval_precisions, 0)
    plt.fill_between(mean_recall, precision_lower, precision_upper, color=color, alpha=.2)


"""
Example of usage:
plt.figure(figsize=(10,10)) #figsize=(10,10)
plot_pr_auc_with_std(test_data, 'XGBoost general', color=(0.00392156862745098, 0.45098039215686275, 0.6980392156862745))
plot_pr_auc_with_std(test_data_xgboost_diagnosis, 'XGBoost per diagnosis', color=(0.8705882352941177, 0.5607843137254902, 0.0196078431372549))
plot_pr_auc_with_std(test_data_baseline_clinical, 'Clinical baseline', color=(0.00784313725490196, 0.6196078431372549, 0.45098039215686275))
plot_pr_auc_with_std(test_data_baseline_diagnosis, 'Diagnosis baseline', color=(0.8352941176470589, 0.3686274509803922, 0.0))
percentage_target = test_data['crisis_in_4_weeks'].mean()
plt.plot([0, 1], [percentage_target, percentage_target], linestyle='--', lw=2, color='black',
            label='Random', alpha=.8)
plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.legend(loc="upper right")
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.savefig('nature_medicine_paper/final_submission/precision_recall_plot.pdf', format='pdf', dpi=400)
plt.show()"""