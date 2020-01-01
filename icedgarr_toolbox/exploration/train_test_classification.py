import random
from typing import List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_predict


def train_test_classification_score(train_data: pd.DataFrame, test_data: pd.DataFrame,
                                    features=List[str]) -> pd.DataFrame:
    """Try to classify train/test samples from total dataframe"""

    minimum_size = min(len(train_data), len(test_data))
    subset_train = train_data.loc[random.sample(list(train_data.index), minimum_size)]
    subset_test = test_data.loc[random.sample(list(test_data.index), minimum_size)]
    # Create a target which is 1 for training rows, 0 for test rows
    subset_train['train'] = 1
    subset_test['train'] = 0
    data = pd.concat([subset_train, subset_test])
    y = data['train'].values
    # Perform shuffled CV predictions of train/test label
    predictions = cross_val_predict(
        RandomForestClassifier(n_estimators=500, n_jobs=4, class_weight="balanced"),
        data[features], y, method='predict_proba',
        cv=StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=42
        )
    )[:, 1]
    # Show the classification report
    print(classification_report(y, np.round(predictions)))
    print(roc_auc_score(y, predictions))
    return pd.DataFrame({'target': y, 'predictions': predictions})
