import pandas as pd
import numpy as np
import scipy


class CompareAUROCDeLongTest:

    def __init__(self, targets, predicted_proba_1, predicted_proba_2):
        self.target_preds = pd.DataFrame({'targets': targets, 'predicted_proba_1': predicted_proba_1,
                                          'predicted_proba_2': predicted_proba_2})

    def compute_z_statistic(self, targets, predicted_proba_1, predicted_proba_2):
        auc_1 = self.compute_auc_mann_whitney(targets, predicted_proba_1)
        auc_2 = self.compute_auc_mann_whitney(targets, predicted_proba_2)
        variance_1 = self.compute_auroc_variance(targets, predicted_proba_1)
        variance_2 = self.compute_auroc_variance(targets, predicted_proba_2)
        covariance = self.compute_auroc_covariance(targets, predicted_proba_1, predicted_proba_2)

        z = (auc_1 - auc_2) / np.sqrt(variance_1 + variance_2 - covariance * np.sqrt(variance_1) * np.sqrt(variance_2))
        return z

    def compute_p_value(self, targets, predicted_proba_1, predicted_proba_2):
        z_statistic = self.compute_z_statistic(targets, predicted_proba_1, predicted_proba_2)
        p_value = scipy.stats.norm.sf(abs(z_statistic)) * 2
        return p_value

    def compute_auc_mann_whitney(self, targets, predicted_probabilities):
        target_pred = pd.DataFrame({'targets': targets, 'predicted_proba': predicted_probabilities})
        target_pred.sort_values(by=['predicted_proba', 'targets'], ascending=False, inplace=True)
        target_pred['num_times'] = (target_pred['targets'] == 1).cumsum()

        num_pos_greater_or_equal_neg = target_pred[target_pred['targets'] == 0]['num_times'].sum()
        num_pos_equal_neg = self._compute_num_pos_equal_neg(target_pred)
        auc = ((num_pos_greater_or_equal_neg - num_pos_equal_neg / 2) /
               ((target_pred['targets'] == 0).sum() * (target_pred['targets'] == 1).sum()))
        return auc

    def compute_auroc_variance(self, targets, predicted_probabilities):
        target_pred = pd.DataFrame({'targets': targets, 'predicted_proba': predicted_probabilities})
        target_pred.sort_values(by='predicted_proba', ascending=False, inplace=True)
        place_value_neg, place_value_pos = self._compute_placement_values(target_pred)
        variance = np.var(place_value_pos) / len(place_value_pos) + np.var(place_value_neg) / len(place_value_neg)
        return variance

    def compute_auroc_covariance(self, targets, predicted_probabilities_1, predicted_probabilities_2):
        auc_1 = self.compute_auc_mann_whitney(targets, predicted_probabilities_1)
        target_pred_1 = pd.DataFrame({'targets': targets, 'predicted_proba': predicted_probabilities_1})
        target_pred_1.sort_values(by='predicted_proba', ascending=False, inplace=True)
        place_value_neg_1, place_value_pos_1 = self._compute_placement_values(target_pred_1)

        auc_2 = self.compute_auc_mann_whitney(targets, predicted_probabilities_2)
        target_pred_2 = pd.DataFrame({'targets': targets, 'predicted_proba': predicted_probabilities_2})
        target_pred_2.sort_values(by='predicted_proba', ascending=False, inplace=True)
        place_value_neg_2, place_value_pos_2 = self._compute_placement_values(target_pred_2)

        cov_value_neg = self._compute_cov_value(auc_1, place_value_neg_1, auc_2, place_value_neg_2)
        cov_value_pos = self._compute_cov_value(auc_1, place_value_pos_1, auc_2, place_value_pos_2)

        covariance = cov_value_neg / len(place_value_neg_1) + cov_value_pos / len(place_value_pos_1)
        return covariance
    
    def compute_conf_interval(self, auroc, auroc_variance, alpha=0.05):
        auroc_std = np.sqrt(auroc_variance)
        lower_upper_q = np.abs(np.array([0, 1]) - (1 - alpha) / 2)

        conf_interval = scipy.stats.norm.ppf(
            lower_upper_q,
            loc=auroc,
            scale=auc_std)

        conf_interval[conf_interval > 1] = 1
        conf_interval[conf_interval < 0] = 0
        return conf_interval

    def _compute_cov_value(self, auc_1, place_value_1, auc_2, place_value_2):
        cov_value = ((place_value_1 - auc_1) * (place_value_2 - auc_2)).sum() / (len(place_value_1) - 1)
        return cov_value

    def _compute_placement_values(self, target_pred):
        # assumes target_pred sorted by predicted_proba and targets in descending order
        target_pred['num_times_neg'] = (target_pred['targets'] == 1).cumsum()
        target_pred = target_pred.sort_values(by='predicted_proba', ascending=True).copy()
        target_pred['num_times_pos'] = (target_pred['targets'] == 0).cumsum()
        target_neg = target_pred[target_pred['targets'] == 0].sort_index()
        target_pos = target_pred[target_pred['targets'] == 1].sort_index()
        place_value_neg = target_neg['num_times_neg'] / (target_pred['targets'] == 1).sum()
        place_value_pos = target_pos['num_times_pos'] / (target_pred['targets'] == 0).sum()
        return place_value_neg.values, place_value_pos.values

    def _compute_num_pos_equal_neg(self, target_pred):
        # assumes target_pred sorted by predicted_proba and targets in descending order
        repeated_targets = target_pred[
            (target_pred['predicted_proba'].shift(1) == target_pred['predicted_proba']) |
            (target_pred['predicted_proba'].shift(-1) == target_pred['predicted_proba'])]
        repeated_prob_with_both_classes = repeated_targets.groupby('predicted_proba')['targets'].nunique()
        repeated_prob_with_both_classes = list(
            repeated_prob_with_both_classes[repeated_prob_with_both_classes == 2].index)

        if repeated_prob_with_both_classes:
            repeated_targets_with_both_classes = repeated_targets[
                repeated_targets['predicted_proba'].isin(list(repeated_prob_with_both_classes))]
            num_pos_equal_neg = \
                repeated_targets_with_both_classes.groupby('predicted_proba')['targets'].value_counts().rename(
                    'counts').reset_index().groupby('predicted_proba')['counts'].prod().sum()
        else:
            num_pos_equal_neg = 0
        return num_pos_equal_neg
