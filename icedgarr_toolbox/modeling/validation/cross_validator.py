from numbers import Number
from typing import List, Callable

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import BaseCrossValidator


class CrossValidator:
    def __init__(self, splitter: BaseCrossValidator, metric: Callable[[np.array, np.array], Number]):
        self.splitter = splitter
        self.metric = metric

    def compute_scores(self, estimator, data: pd.DataFrame, feature_columns: List[str],
                       target_column: List[str]) -> (List[Number], List[DataFrame]):
        data = data.reset_index(drop=True)
        scores = []
        predictions = []
        data_test = []
        for train_id, test_id in self.splitter.split(data, data[target_column]):
            train_x, train_y = data.loc[train_id, feature_columns], data.loc[train_id, target_column]

            test_x, test_y = data.loc[test_id, feature_columns], data.loc[test_id, target_column]

            estimator.fit(train_x, train_y)
            predicted_values = estimator.predict(test_x)
            scores.append(self.metric(test_y, predicted_values))
            predictions_dataframe = pd.DataFrame({'predicted_values': predicted_values})
            predictions_dataframe[target_column] = test_y.values
            predictions.append(predictions_dataframe)
            data_test.append(data.loc[test_id])

        return scores, predictions, data_test
