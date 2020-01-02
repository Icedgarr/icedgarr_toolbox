from numpy import array

from icedgarr_toolbox.modeling.base import Model


class MeanPredictor(Model):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = None

    def fit(self, x, y):
        self.mean = y.mean()

    def predict(self, x) -> array:
        return array([self.mean] * len(x))
