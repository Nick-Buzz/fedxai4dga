import tensorflow as tf
from keras_tuner import RandomSearch, Hyperband, GridSearch, HyperModel
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import pandas as pd
from .ModelBase import ModelBase

base_path = r"C:\Users\nbazo\Desktop\Netmode\fedxai4dga\fedxai4dga"


class MLPAttentionModel2(ModelBase):
    def __init__(self, dropout_rate=0.2, activation='relu',
                 optimizer='adam', loss='binary_crossentropy',
                 num_heads=8, key_dim=50):
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.model = None

    def build(self, features_number):
        # Define the inputs
        inputs = Input(shape=(features_number,))
        x_reshaped = tf.expand_dims(inputs, axis=1)
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim
        )(x_reshaped, x_reshaped)

        # Remove the added dimension
        attention_output = tf.squeeze(attention_output, axis=1)

        # Concatenate the attention output with the MLP output
        combined = Concatenate()([inputs, attention_output])

        # Define the second MLP module
        x = Dense(500, activation=self.activation)(combined)
        x = BatchNormalization()(x)
        #x = Dropout(self.dropout_rate)(x)

        x = Dense(250, activation=self.activation)(x)
        x = BatchNormalization()(x)
        #x = Dropout(self.dropout_rate)(x)

        x = Dense(100, activation=self.activation)(x)
        x = BatchNormalization()(x)

        outputs = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=['accuracy'])
        # return self.model

    def fit(self, X, y, **kwargs):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        history = self.model.fit(X, y, validation_split=0.2, epochs=10, batch_size=512,
                                 callbacks=[early_stopping], **kwargs)
        return history

    def score(self, X, y, sample_weight=None):

        return accuracy_score(y, self.model.predict(X), sample_weight=sample_weight)

    def get_params(self, deep=True):
        return {
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "optimizer": self.optimizer,
            "loss": self.loss,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim
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

    def tune(self, X, y, algorithm="RandomSearch", epochs=5, export_csv=False, **kwargs):
        features_number = X.shape[1]  # Get number of features from input data

        class MLPHyperModel(HyperModel):
            def build(self, hp):
                # Define inputs
                inputs = Input(shape=(features_number,))
                x_reshaped = tf.expand_dims(inputs, axis=1)

                # Tune attention parameters
                num_heads = hp.Int('num_heads', min_value=4, max_value=16, step=4)
                key_dim = hp.Int('key_dim', min_value=32, max_value=128, step=32)

                attention_output = MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=key_dim
                )(x_reshaped, x_reshaped)

                attention_output = tf.squeeze(attention_output, axis=1)
                combined = Concatenate()([inputs, attention_output])

                # Tune MLP parameters
                units_1 = hp.Int('units_1', min_value=200, max_value=600, step=100)
                units_2 = hp.Int('units_2', min_value=100, max_value=300, step=50)
                units_3 = hp.Int('units_3', min_value=50, max_value=150, step=25)

                activation = hp.Choice('activation', values=['relu', 'tanh'])
                dropout_rate = hp.Float('dropout_rate', 0.1, 0.5, step=0.1)

                # Build MLP layers
                x = Dense(units_1, activation=activation)(combined)
                x = BatchNormalization()(x)
                x = Dropout(dropout_rate)(x)

                x = Dense(units_2, activation=activation)(x)
                x = BatchNormalization()(x)
                x = Dropout(dropout_rate)(x)

                x = Dense(units_3, activation=activation)(x)
                x = BatchNormalization()(x)

                outputs = Dense(1, activation='sigmoid')(x)

                model = Model(inputs=inputs, outputs=outputs)

                optimizer = hp.Choice('optimizer', values=['adam', 'sgd'])
                model.compile(
                    optimizer=optimizer,
                    loss='binary_crossentropy',
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
            tuner_params["max_trials"] = kwargs.get('max_trials', 3)

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
