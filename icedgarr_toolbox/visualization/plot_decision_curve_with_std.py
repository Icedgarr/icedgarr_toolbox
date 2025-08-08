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
