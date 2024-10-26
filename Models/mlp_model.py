import pandas as pd
import tensorflow as tf
from keras_tuner import RandomSearch, Hyperband, GridSearch, HyperModel
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score

base_path = r"C:\Users\nbazo\Desktop\Netmode\fedxai4dga\fedxai4dga"

"""
def build_model(hidden_layers, dropout_rate, activation, optimizer, loss):
    model = tf.keras.models.Sequential()
    for units in hidden_layers:
        model.add(tf.keras.layers.Dense(units, activation=activation))
        model.add(tf.keras.layers.Dropout(dropout_rate))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    return model
"""

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
        self.model = tf.keras.models.Sequential()
        for units in self.hidden_layers:
            self.model.add(tf.keras.layers.Dense(units, activation=self.activation))
            self.model.add(tf.keras.layers.Dropout(self.dropout_rate))
        self.model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
        self.model.compile(loss=self.loss, optimizer=self.optimizer, metrics=['accuracy'])

    def fit(self, X, y, **kwargs):
        if self.model is None:
            self.build()
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        history = self.model.fit(X, y, validation_split=0.2, epochs=100, batch_size=512,
                                 callbacks=[early_stopping], **kwargs)
        return history

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

    def tune(self, X, y, algorithm="RandomSearch", epochs=50, export_csv=False, **kwargs):
        class MLPHyperModel(HyperModel):
            def build(self, hp):
                # Create model directly without external function
                hidden_layers = [hp.Int('units_' + str(i), min_value=100, max_value=500, step=100)
                                 for i in range(3)]
                hidden_layers.sort(reverse=True)  # Sort in descending order

                model = tf.keras.models.Sequential()
                for units in hidden_layers:
                    model.add(tf.keras.layers.Dense(units,
                                                    activation=hp.Choice('activation', values=['relu', 'tanh'])))
                    model.add(tf.keras.layers.Dropout(
                        hp.Float('dropout_rate', 0.1, 0.5, step=0.1)))
                model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                model.compile(
                    loss='binary_crossentropy',
                    optimizer=hp.Choice('optimizer', values=['adam', 'sgd']),
                    metrics=['accuracy']
                )
                return model

        tuner_classes = {
            "RandomSearch": RandomSearch,
            "Hyperband": Hyperband,
            "GridSearch": GridSearch
        }

        if algorithm not in tuner_classes:
            raise ValueError(f"Unsupported algorithm. Choose from {', '.join(tuner_classes.keys())}")

        tuner_class = tuner_classes[algorithm]
        tuner_params = {
            "hypermodel": MLPHyperModel(),
            "objective": 'val_accuracy',
            "directory": f"{base_path}/Results/mlp/{algorithm}/",
            "project_name": f'mlp_hyperparameter_tuning_{algorithm}',
            **kwargs
        }

        if algorithm == "Hyperband":
            tuner_params["max_epochs"] = epochs
        else:
            tuner_params["max_trials"] = kwargs.get('max_trials', 10)

        tuner = tuner_class(**tuner_params)

        tuner.search(X, y, epochs=epochs, validation_split=0.2)

        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

        if export_csv:
            trials = tuner.oracle.get_best_trials()
            results_df = pd.DataFrame([
                {
                    'trial': trial.trial_id,
                    'loss': trial.score,
                    **trial.hyperparameters.values
                }
                for trial in trials
            ])
            results_df.to_csv(f'{base_path}/Results/mlp/{algorithm}/tuning_results.csv', index=False)

        # Convert HyperParameters to a dictionary compatible with MLPModel
        best_params = {
            "hidden_layers": [best_hp.get(f'units_{i}') for i in range(3)],
            "dropout_rate": best_hp.get('dropout_rate'),
            "activation": best_hp.get('activation'),
            "optimizer": best_hp.get('optimizer'),
            "loss": 'binary_crossentropy'
        }

        return best_params

# Example usage:
# model = MLPModel()
# best_params = model.tune(X, y, algorithm="RandomSearch", epochs=50, export_csv=True)
# model.set_params(**best_params)
# model.fit(X, y)
