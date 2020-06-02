from abc import ABC, abstractmethod
from typing import Any, NoReturn


class AbstractBaseModel(ABC):
    """ Abstract class to define methods to implement
    for a Model class inspired by Scikit Learn API.
    """

    def __init__(self):
        self._is_fitted = False
        self.X_train = None

    @abstractmethod
    def fit(self, series) -> NoReturn:
        """ Fit a model """
        self._is_fitted = True
        self.X_train = series

    @abstractmethod
    def predict(self, horizon) -> Any:
        """ Usage of the model to predict values """
        if self._is_fitted is False:
            raise Exception('fit() must be called before predict()')
