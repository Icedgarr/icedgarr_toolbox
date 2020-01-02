from abc import ABCMeta, abstractmethod


class Model(metaclass=ABCMeta):

    def __init__(self, *args, **kwargs):
        pass

    @abstractmethod
    def fit(self, x, y):
        pass

    @abstractmethod
    def predict(self, x):
        pass
