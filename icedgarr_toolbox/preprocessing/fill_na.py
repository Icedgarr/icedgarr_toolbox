from pandas import DataFrame


def fillna_median(data: DataFrame) -> DataFrame:
    return data.fillna(data.median(axis=0))


def fillna_mean(data: DataFrame) -> DataFrame:
    return data.fillna(data.median(axis=0))
