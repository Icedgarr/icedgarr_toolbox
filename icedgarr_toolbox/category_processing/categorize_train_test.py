import pandas as pd

from icedgarr_toolbox.category_processing.code_categorical_features import code_categorical_features


def categorize_train_and_test(data_train, data_test, categorical_columns):
    data_train['train'] = 1
    data_test['train'] = 0
    data_all = pd.concat([data_train, data_test], sort=False).reset_index(drop=True)
    data_all = code_categorical_features(categorical_columns=categorical_columns, dataframe=data_all)
    data_train = data_all.loc[data_all['train'] == 1]
    data_test = data_all.loc[data_all['train'] == 0]
    return data_train, data_test
