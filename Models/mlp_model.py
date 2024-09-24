import pandas as pd
import tensorflow as tf
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch, Hyperband
from sklearn.base import BaseEstimator, ClassifierMixin
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
        """
        Perform hyperparameter tuning using Keras Tuner.

        Parameters:
        algorithm (str): The tuning algorithm to use ('RandomSearch' or 'Hyperband').
        epochs (int): The number of epochs for training.
        export_csv (bool): Flag to export results to CSV.
        **kwargs: Additional arguments to pass to the tuner.

        Returns:
        best_hyperparams: The best hyperparameters found during tuning.
        """

        class MLPHyperModel(HyperModel):
            def build(self, hp):
                hidden_layers = [hp.Int('units_' + str(i), min_value=100, max_value=500, step=100) for i in range(3)].sort()
                dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)
                activation = hp.Choice('activation', values=['relu', 'tanh'])
                optimizer = hp.Choice('optimizer', values=['adam', 'sgd'])

                model = MLPModel(hidden_layers=hidden_layers, dropout_rate=dropout_rate,
                                 activation=activation, optimizer=optimizer)
                model.build()
                return model.model

        # Instantiate the tuner based on the specified algorithm
        if algorithm == "RandomSearch":
            tuner = RandomSearch(
                MLPHyperModel(),
                objective='val_accuracy',
                max_trials=10,
                directory='my_dir',
                project_name='mlp_hyperparameter_tuning',
                **kwargs
            )
        elif algorithm == "Hyperband":
            tuner = Hyperband(
                MLPHyperModel(),
                objective='val_accuracy',
                max_epochs=epochs,
                directory='my_dir',
                project_name='mlp_hyperparameter_tuning',
                **kwargs
            )
        else:
            raise ValueError("Unsupported algorithm. Choose 'RandomSearch' or 'Hyperband'.")

        # Perform the hyperparameter search
        tuner.search(X, y, epochs=epochs, validation_split=0.2)

        # Retrieve and return the best hyperparameters
        best_hyperparams = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Export results to CSV if required
        if export_csv:
            trials = tuner.oracle.get_best_trials(num_trials=tuner.oracle.trials_count)
            results_df = pd.DataFrame([
                {
                    'trial': trial.trial_id,
                    'loss': trial.score,
                    **trial.hyperparameters.values
                }
                for trial in trials
            ])
            results_df.to_csv('tuning_results.csv', index=False)

        return best_hyperparams
