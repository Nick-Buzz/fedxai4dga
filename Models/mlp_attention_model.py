from datetime import datetime

import tensorflow as tf
from keras_tuner import RandomSearch, Hyperband, GridSearch, HyperModel
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import pandas as pd
import os

from .ModelBase import ModelBase


base_path = r"C:\Users\nbazo\Desktop\Netmode\fedxai4dga\fedxai4dga"


class MLPAttentionModel(ModelBase):
    def __init__(self, dropout_rate=0.2, activation='relu',
                 optimizer='adam', loss='binary_crossentropy',
                 num_heads=8, key_dim=200,
                 hidden_layers_1=[300, 200, 200],
                 hidden_layers_2=[300, 200]):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.hidden_layers_1 = hidden_layers_1
        self.hidden_layers_2 = hidden_layers_2
        self.model = None

    def build(self, features_number):
        # Define the inputs
        inputs = Input(shape=(features_number,))

        # First MLP module
        x = inputs
        for i, units in enumerate(self.hidden_layers_1):
            x = Dense(units,
                      activation=self.activation if i < len(self.hidden_layers_1) - 1 else 'linear')(x)
            x = Dropout(self.dropout_rate)(x)

        # Reshape for the MultiHeadAttention layer
        x_reshaped = tf.expand_dims(x, axis=1)

        # Self-attention layer
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim
        )(x_reshaped, x_reshaped)

        # Remove the added dimension
        attention_output = tf.squeeze(attention_output, axis=1)

        # Concatenate the attention output with the MLP output
        combined = Concatenate()([x, attention_output])

        # Second MLP module
        x = combined
        for units in self.hidden_layers_2:
            x = Dense(units, activation=self.activation)(x)

        outputs = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=['accuracy'])

    def fit(self, X, y, **kwargs):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        history = self.model.fit(X, y, validation_split=0.2, epochs=50, batch_size=1024,
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
            "key_dim": self.key_dim,
            "hidden_layers_1": self.hidden_layers_1,
            "hidden_layers_2": self.hidden_layers_2
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

    def tune(self, X, y, algorithm="GridSearch", epochs=25, export_csv=True, **kwargs):
        features_number = X.shape[1]

        class MLPAttentionHyperModel(HyperModel):
            def build(self, hp):
                # Define inputs
                inputs = Input(shape=(features_number,))

                # First MLP module - Tune architecture
                x = inputs
                hidden_layers_1 = [
                    hp.Int(f'units_1_{i}', min_value=50, max_value=400, step=50)
                    for i in range(3)
                ]
                hidden_layers_1.sort(reverse=True)  # Sort in descending order

                for i, units in enumerate(hidden_layers_1):
                    x = Dense(
                        units,
                        activation='relu'
                        if i < len(hidden_layers_1) - 1 else 'linear'
                    )(x)
                    x = Dropout(hp.Float('dropout_rate', 0.1, 0.5, step=0.1))(x)

                # Attention mechanism - Tune parameters
                x_reshaped = tf.expand_dims(x, axis=1)
                num_heads = hp.Int('num_heads', min_value=2, max_value=16, step=2)
                key_dim = hp.Int('key_dim', min_value=100, max_value=300, step=50)

                attention_output = MultiHeadAttention(
                    num_heads=num_heads,
                    key_dim=key_dim
                )(x_reshaped, x_reshaped)

                attention_output = tf.squeeze(attention_output, axis=1)
                combined = Concatenate()([x, attention_output])

                # Second MLP module - Tune architecture
                x = combined
                hidden_layers_2 = [
                    hp.Int(f'units_2_{i}', min_value=50, max_value=400, step=50)
                    for i in range(2)
                ]
                hidden_layers_2.sort(reverse=True)

                for units in hidden_layers_2:
                    x = Dense(units, activation='relu')(x)

                outputs = Dense(1, activation='sigmoid')(x)

                model = Model(inputs=inputs, outputs=outputs)
                optimizer = hp.Choice('optimizer', values=['adam', 'sgd'])

                model.compile(
                    optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                return model

            def fit(self, hp, model, *args, **kwargs):
                # Define batch size as a hyperparameter
                batch_size = hp.Choice('batch_size', values=[256, 512, 1024])

                # Update kwargs with the batch_size
                kwargs['batch_size'] = batch_size

                return model.fit(*args, **kwargs)

        tuner_classes = {
            "RandomSearch": RandomSearch,
            "Hyperband": Hyperband,
            "GridSearch": GridSearch
        }

        if algorithm not in tuner_classes:
            raise ValueError(f"Unsupported algorithm. Choose from {', '.join(tuner_classes.keys())}")

        tuner_class = tuner_classes[algorithm]
        tuner_params = {
            "hypermodel": MLPAttentionHyperModel(),
            "objective": 'accuracy',
            "directory": f"{base_path}/Results/mlp-attention/{algorithm}/",
            "project_name": f'mlp_attention_hyperparameter_tuning_{algorithm}'
                            f'{datetime.now().strftime("%Y-%m-%d_%H%M%S")}',
            **kwargs
        }

        if algorithm == "Hyperband":
            tuner_params["max_epochs"] = epochs
        else:
            tuner_params["max_trials"] = kwargs.get('max_trials', 20)

        tuner = tuner_class(**tuner_params)
        tuner.search(X, y, epochs=epochs, validation_split=0.2)
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

        if export_csv:
            # Get all trials instead of just the top 5
            all_trials = tuner.oracle.trials

            # Create a list to store all trial data
            trial_data = []

            for trial_id, trial in all_trials.items():

                # Create a dictionary with trial information
                trial_info = {
                    'trial_id': trial_id,
                    'score': trial.score,  # validation accuracy
                    'loss': trial.metrics.get_last_value('val_loss'),
                    # 'epochs_trained': len(trial.metrics.get('loss', [])),
                    'status': trial.status,
                }

                # Add all hyperparameters
                trial_info.update(trial.hyperparameters.values)

                trial_data.append(trial_info)

            # Convert to DataFrame and save
            results_df = pd.DataFrame(trial_data)

            # Sort by score (validation accuracy) in descending order
            results_df = results_df.sort_values('score', ascending=False)

            # Save to CSV
            results_path = f'{base_path}/Results/mlp-attention/{algorithm}/'
            os.makedirs(results_path, exist_ok=True)
            results_df.to_csv(f'{results_path}/'
                              f'all_trials_results{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.csv',
                              index=False)

            # Also save a summary of the best trial
            best_trial = results_df.iloc[0].to_dict()
            pd.DataFrame([best_trial]).to_csv(f'{results_path}/'
                                              f'best_trial_results{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.csv',
                                              index=False)

        # Convert HyperParameters to a dictionary compatible with MLPAttentionModel
        best_params = {
            "hidden_layers_1": [best_hp.get(f'units_1_{i}') for i in range(3)],
            "hidden_layers_2": [best_hp.get(f'units_2_{i}') for i in range(2)],
            "dropout_rate": best_hp.get('dropout_rate'),
            "optimizer": best_hp.get('optimizer'),
            "loss": 'binary_crossentropy',
            "num_heads": best_hp.get('num_heads'),
            "key_dim": best_hp.get('key_dim'),
            "batch_size": best_hp.get('batch_size')
        }

        return best_params

