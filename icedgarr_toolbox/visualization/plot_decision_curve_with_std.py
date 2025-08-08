import numpy as np
import matplotlib.pyplot as plt

from icedgarr_toolbox.metrics.compute_net_benefit import compute_net_benefit


def plot_decision_curve_with_std(data, model_name, color, prediction_column='calibrated_predictions',
                                 original_prediction_column='predictions', target_column='crisis_in_4_weeks',
                                 net_benefit_used='net_benefit_treated', split_column='week_number'):
    probs_to_plot = list(np.linspace(0, 0.5, 100))
    net_benefit_to_plot = []
    for week in data[split_column].unique():
        net_benefit = [
            compute_net_benefit(data[data[split_column] == week], p, prediction_column=prediction_column,
                                target_column=target_column)[net_benefit_used] for p in probs_to_plot]

        net_benefit_to_plot.append(net_benefit)

    mean_net_benefit = np.mean(net_benefit_to_plot, axis=0)
    plt.plot(probs_to_plot, mean_net_benefit, color=color,
             label=r'%s' % (model_name),
             lw=2, alpha=.8)

    std_net_benefit = np.std(net_benefit_to_plot, axis=0)
    net_benefit_upper = np.minimum(mean_net_benefit + std_net_benefit, 1)
    net_benefit_lower = np.maximum(mean_net_benefit - std_net_benefit, 0)
    plt.fill_between(probs_to_plot, net_benefit_lower, net_benefit_upper, color=color, alpha=.2)


"""
Example usage:
%%time
probs_to_plot = list(np.linspace(0, 0.5, 100))
net_benefit_to_plot = [compute_net_benefit(test_data, p)['net_benefit_treated'] for p in probs_to_plot]
prevalence = test_data['crisis_in_4_weeks'].mean()
treat_all_patients = [prevalence -(1-prevalence) * (p/(1-p)) for p in probs_to_plot]

plt.figure(figsize=(10,10)) #figsize=(10,10)
#plt.plot(probs_to_plot, net_benefit_to_plot)
plt.plot([0, 0.5], [prevalence, prevalence], linestyle='--', lw=2, color='black',
            label='Perfect classifier', alpha=.8)
plt.plot([0, 0.5], [0, 0], linestyle=':', lw=2, color='black',
            label='Treat no patients', alpha=.8)
plt.plot(probs_to_plot, treat_all_patients, linestyle='-.', lw=2, color='black',
            label='Treat all patients', alpha=.8)

plot_decision_curve_with_std(test_data, 'XGBoost general', 
                             color=(0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
                             prediction_column='calibrated_predictions', 
                             net_benefit_used='net_benefit_treated')

plot_decision_curve_with_std(test_data,'XGBoost per diagnosis', 
                             color=(0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
                             prediction_column='calibrated_predictions_per_diagnosis', 
                             net_benefit_used='net_benefit_treated')

plot_decision_curve_with_std(test_data, 'Clinical baseline', 
                             color=(0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
                             prediction_column='calibrated_predictions_doctor_baseline', 
                             net_benefit_used='net_benefit_treated')

plot_decision_curve_with_std(test_data, 'Diagnosis baseline', 
                             color=(0.8352941176470589, 0.3686274509803922, 0.0),
                             prediction_column='calibrated_predictions_diagnosis_baseline', 
                             net_benefit_used='net_benefit_treated')

plt.xlim(0, 0.5)
plt.xticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5], labels=[0, 10, 20, 30, 40, 50])
plt.ylim(-0.001, prevalence+0.001)
plt.xlabel('Threshold probabilities (%)')
plt.ylabel('Net benefit')
plt.legend()
plt.tight_layout()
plt.savefig('nature_medicine_paper/final_submission/decision_curve_benefit.pdf', format='pdf', dpi=400)
plt.xlim(0, 0.5)
plt.ylim(-0.001, prevalence+0.001)
plt.show()"""