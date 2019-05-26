from typing import List

from pandas import DataFrame


def code_categorical_features(categorical_columns: List[str], dataframe: DataFrame) -> DataFrame:
    dataframe[categorical_columns] = dataframe[categorical_columns].astype('category')
    category_names = {}
    for category in categorical_columns:
        mylist = []
        subcategory_names = {'null': -1}
        for cat, index in enumerate(dataframe[category].cat.categories):
            subcategory_names[index] = cat
        mylist.append(subcategory_names)
        category_names[category] = mylist
        dataframe[category] = dataframe[category].cat.codes
    return dataframe
