from typing import List

import pandas as pd
from scipy.stats import ks_2samp


def compare_train_test_distributions(train_data: pd.DataFrame, test_data: pd.DataFrame,
                                     features: List[str]) -> pd.DataFrame:
    """
    If the K-S statistic is small or the p-value is high, then we cannot
    reject the hypothesis that the distributions of the two samples
    are the same. (If p-value is high the distributions are probably the same, if p-value is small they are different)
    """
    p_values_list = []
    for feature in features:
        compute_ks_test = ks_2samp(train_data[feature], test_data[feature])
        p_values = {'feature': feature, 'p_value': compute_ks_test.pvalue}
        p_values_list.append(p_values)
    p_values_df = pd.DataFrame(p_values_list).sort_values('p_value')
    return p_values_df
