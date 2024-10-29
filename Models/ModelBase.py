from abc import ABC, abstractmethod

from sklearn.base import ClassifierMixin, BaseEstimator
import tensorflow as tf


class ModelBase(ABC,BaseEstimator, ClassifierMixin, tf.keras.Model):

    @abstractmethod
    def build(self,**kwargs):
        pass

    @abstractmethod
    def fit(self, X_train, y_train):
        pass

