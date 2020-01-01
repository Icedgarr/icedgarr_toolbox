from typing import List

from pandas import DataFrame

DEFAULT_STATS = ['min', 'max', 'mean', 'median', 'std']


def create_statistics(data: DataFrame, columns: List[str], prefix: str, stats: List[str] = DEFAULT_STATS) -> DataFrame:
    print(prefix)
    statistics_features = data[columns].agg(stats, axis=1)
    statistics_features['perc_positive'] = (data[columns] > 0).mean(axis=1)
    statistics_features['perc_nan'] = (data[columns].isnull()).mean(axis=1)
    return statistics_features.rename(columns={col: f'{prefix}_{col}' for col in statistics_features.columns})
