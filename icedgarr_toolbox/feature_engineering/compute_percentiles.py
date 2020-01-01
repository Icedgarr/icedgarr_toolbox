from typing import List

import pandas as pd


def compute_percentiles_list_columns(data: pd.DataFrame, list_columns: List[str]) -> pd.DataFrame:
    percentiles_df = pd.DataFrame(index=data.index)
    for column in list_columns:
        percentiles_df[f'{column}_percentile'] = compute_linear_interpolation_percentile_column(data[column])
    return percentiles_df


def compute_linear_interpolation_percentile_column(data: pd.Series) -> pd.Series:
    size = data.size - 1
    return data.rank(method='max').apply(lambda x: (x - 1) / size)
