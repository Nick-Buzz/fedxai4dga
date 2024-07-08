import numpy as np
import xgboost as xgb
from .ModelBase import ModelBase


class XGBoostModel(ModelBase):
    def __init__(self):
        self.model = None

    def build(self):
        rng = np.random.RandomState(42)
        self.model = xgb.XGBClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=rng)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)