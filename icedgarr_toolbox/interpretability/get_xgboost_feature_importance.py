import pandas as pd


def get_xgboost_feature_importance(estimator, feature_columns):
    return pd.DataFrame({'feature': feature_columns, 'importance': estimator.feature_importances_})\
        .sort_values('importance', ascending=False).set_index('feature')
