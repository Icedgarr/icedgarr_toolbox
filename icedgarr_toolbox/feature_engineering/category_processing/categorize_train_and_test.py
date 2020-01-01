from typing import List

import pandas as pd

from icedgarr_toolbox.feature_engineering.category_processing.code_categorical_features import code_categorical_features


def categorize_train_and_test(data_train: pd.DataFrame, data_test: pd.DataFrame,
                              categorical_columns: List[str]) -> (pd.DataFrame, pd.DataFrame):
    data_train['train'] = 1
    data_test['train'] = 0
    data_all = pd.concat([data_train, data_test], sort=False).reset_index(drop=True)
    data_all = code_categorical_features(categorical_columns=categorical_columns, dataframe=data_all)
    data_train = data_all.loc[data_all['train'] == 1]
    data_test = data_all.loc[data_all['train'] == 0]
    return data_train, data_test
