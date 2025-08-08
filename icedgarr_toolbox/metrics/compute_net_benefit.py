def net_benefit_treated(true_positive_rate, false_positive_rate, p):
    return true_positive_rate - false_positive_rate * (p / (1-p))

def net_benefit_untreated(true_negative_rate, false_negative_rate, p):
    return true_negative_rate - false_negative_rate * (p / (1-p))

def net_benefit_overall(net_benefit_treated, net_benefit_untreated):
    return net_benefit_treated + net_benefit_untreated

def compute_net_benefit(data, p, prediction_column='calibrated_predictions', target_column='crisis_in_4_weeks'):
    true_positive_rate = ((data[prediction_column] >= p) & (data[target_column] == 1)).mean()
    false_positive_rate = ((data[prediction_column] >= p) & (data[target_column] == 0)).mean()
    true_negative_rate = ((data[prediction_column] < p) & (data[target_column] == 0)).mean()
    false_negative_rate = ((data[prediction_column] < p) & (data[target_column] == 1)).mean()
    net_benefit_treated_value = net_benefit_treated(true_positive_rate, false_positive_rate, p)
    net_benefit_untreated_value = net_benefit_untreated(true_negative_rate, false_negative_rate, p)
    net_benefit_overall_value = net_benefit_overall(net_benefit_treated_value, net_benefit_untreated_value)
    return {'net_benefit_treated': net_benefit_treated_value, 'net_benefit_untreated': net_benefit_untreated_value,
            'net_benefit_overall': net_benefit_overall_value}