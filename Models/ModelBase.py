from abc import ABC, abstractmethod


class ModelBase(ABC):
    @abstractmethod
    def build(self,**kwargs):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass
