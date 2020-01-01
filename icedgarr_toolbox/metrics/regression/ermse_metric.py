import numpy as np
from sklearn.metrics import mean_squared_error


def ermse_metric(y_actual, y_predicted):
    return np.exp(-np.sqrt(mean_squared_error(y_actual, y_predicted)))
