from abc import ABC, abstractmethod

from sklearn.base import ClassifierMixin, BaseEstimator


class ModelBase(ABC,BaseEstimator, ClassifierMixin):

    @abstractmethod
    def build(self,**kwargs):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

