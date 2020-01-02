import random

import numpy as np


class XGBoostRandomSearch:
    def sample_space(self, num_samples):
        return [self.sample() for _ in range(num_samples)]

    @staticmethod
    def sample():
        parameter_space = {
            'learning_rate': random.choice(np.geomspace(1e-2, 1)),
            'max_depth': random.choice(range(1, 10)),
            'gamma': random.choice(np.geomspace(1e-2, 1)),
            'min_child_weight': random.choice(range(1, 10)),
            'n_estimators': random.choice(range(30, 300)),
            'reg_alpha': random.choice(np.linspace(0.2, 1)),
            'reg_lambda': random.choice(np.linspace(0.2, 2)),
            'colsample_bytree': random.choice(np.linspace(0.1, 1)),
            'njobs': -1,
            'objective': 'reg:squarederror'
        }
        return parameter_space
