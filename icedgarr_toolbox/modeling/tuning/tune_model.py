from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from icedgarr_toolbox.modeling.base import Model
from icedgarr_toolbox.modeling.validation.cross_validator import CrossValidator


class TuneModel:
    def __init__(self, cross_validator: CrossValidator, parameter_spaces: List[dict]):
        self.cross_validator = cross_validator
        self.parameter_spaces = parameter_spaces

    def tune_model(self, model_class: Model, data, feature_columns, target_column):
        scores_dicts = []
        for params in tqdm(self.parameter_spaces):
            print(params)
            model = model_class(**params)
            scores, _, _ = self.cross_validator.compute_scores(model, data, feature_columns, target_column)
            mean_scores, std_scores = np.mean(scores), np.std(scores)
            scores_dicts.append(dict(params, mean_scores=mean_scores, std_scores=std_scores, scores_list=scores))
        return pd.DataFrame(scores_dicts).sort_values('mean_scores', ascending=False)
