from typing import List, Callable
from numbers import Number

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import BaseCrossValidator


def stratified_cross_validation(data: pd.DataFrame, splitter: BaseCrossValidator, estimator,
                                metric: Callable[[np.array, np.array], Number], feature_columns: List[str],
                                target_column: List[str]) -> (List[Number], List[DataFrame]):
    scores = []
    predictions = []
    data_test = []
    for train_id, test_id in splitter.split(data, data[target_column]):
        train_x, train_y = data.loc[train_id, feature_columns], data.loc[train_id, target_column]

        test_x, test_y = data.loc[test_id, feature_columns], data.loc[test_id, target_column]

        estimator.fit(train_x, train_y)
        predicted_values = estimator.predict(test_x)
        predicted_probabilities = estimator.predict_proba(test_x)[:, 1]
        scores.append(metric(test_y, predicted_values))
        predictions_dataframe = pd.DataFrame({'predicted_values': predicted_values,
                                              'predicted_probabilities': predicted_probabilities})
        predictions_dataframe[target_column] = test_y.values
        predictions.append(predictions_dataframe)
        data_test.append(data.loc[test_id])

    return scores, predictions, data_test
