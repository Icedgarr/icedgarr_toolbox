from typing import List

from pandas import DataFrame
from sklearn.model_selection import BaseCrossValidator


def cv_submission(data_train: DataFrame, data_test: DataFrame, splitter: BaseCrossValidator,
                  estimator, feature_columns: List[str], target_column: List[str]):
    i = 1
    for train_id, test_id in splitter.split(data_train, data_train[target_column]):
        train_x, train_y = data_train.loc[train_id, feature_columns], data_train.loc[train_id, target_column]

        test_x, test_y = data_train.loc[test_id, feature_columns], data_train.loc[test_id, target_column]

        estimator.fit(train_x, train_y, eval_set=[(train_x, train_y), (test_x, test_y)])
        data_test[f'predicted_{target_column}_{i}'] = estimator.predict(data_test[feature_columns])
        i += 1
    return data_test
