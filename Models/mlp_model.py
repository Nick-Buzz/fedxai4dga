import tensorflow as tf
from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.metrics import accuracy_score


class MLPModel(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_layers=[300, 200, 200], dropout_rate=0.2, activation='relu',
                 optimizer='adam', loss='binary_crossentropy'):
        self.hidden_layers = hidden_layers
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.model = None

    def build(self):
        model = tf.keras.models.Sequential()
        for units in self.hidden_layers:
            model.add(tf.keras.layers.Dense(units, activation=self.activation))
            model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])
        self.model = model

    def fit(self, X, y, **kwargs):
        if self.model is None:
            self.build()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        self.model.fit(X, y, validation_split=0.2, epochs=1, batch_size=512,
                       callbacks=[early_stopping], **kwargs)
        return self

    def predict(self, X):
        return (self.model.predict(X) > 0.5).astype(int).flatten()

    def score(self, X, y, sample_weight=None):
        return accuracy_score(y, self.predict(X), sample_weight=sample_weight)

    def get_params(self, deep=True):
        return {
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "optimizer": self.optimizer,
            "loss": self.loss
        }

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def __getstate__(self):
        state = self.__dict__.copy()
        if self.model is not None:
            state['model_config'] = self.model.get_config()
            state['model_weights'] = self.model.get_weights()
            del state['model']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'model_config' in state:
            self.model = tf.keras.models.Sequential.from_config(state['model_config'])
            self.model.set_weights(state['model_weights'])


