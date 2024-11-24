import os
from datetime import datetime
import tensorflow as tf
from keras_tuner import RandomSearch, Hyperband, GridSearch, HyperModel
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
import pandas as pd
from .ModelBase import ModelBase

base_path = r"C:\Users\nbazo\Desktop\Netmode\fedxai4dga\fedxai4dga"


class MLPAttentionModel2(ModelBase):
    """
    A neural network model combining Multi-Head Attention with MLP layers for classification tasks.

    This model architecture consists of:
    1. Multi-Head Attention layer for feature interaction learning
    2. Three Dense layers with batch normalization
    3. Final sigmoid layer for binary classification

    Attributes:
        dropout_rate (float): Dropout rate for regularization
        activation (str): Activation function for dense layers
        optimizer (str): Optimizer for model training
        loss (str): Loss function for model training
        num_heads (int): Number of attention heads
        key_dim (int): Dimension of key in attention mechanism
        model (tf.keras.Model): The compiled Keras model
    """

    def __init__(self, dropout_rate=0.2, activation='relu',
                 optimizer='adam', loss='binary_crossentropy',
                 num_heads=8, key_dim=50):
        """
        Initialize the MLPAttentionModel2.

        Args:
            dropout_rate (float, optional): Dropout rate. Defaults to 0.2.
            activation (str, optional): Activation function. Defaults to 'relu'.
            optimizer (str, optional): Optimizer name. Defaults to 'adam'.
            loss (str, optional): Loss function name. Defaults to 'binary_crossentropy'.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            key_dim (int, optional): Key dimension for attention. Defaults to 50.
        """
        super().__init__()
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.model = None

    def build(self, features_number):
        """
        Build the neural network architecture.

        Args:
            features_number (int): Number of input features.
        """
        # Define the inputs
        inputs = Input(shape=(features_number,))
        x_reshaped = tf.expand_dims(inputs, axis=1)

        # Apply attention mechanism
        attention_output = MultiHeadAttention(
            num_heads=self.num_heads,
            key_dim=self.key_dim
        )(x_reshaped, x_reshaped)

        attention_output = tf.squeeze(attention_output, axis=1)
        combined = Concatenate()([inputs, attention_output])

        # MLP layers with batch normalization
        x = Dense(500, activation=self.activation)(combined)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(250, activation=self.activation)(x)
        x = BatchNormalization()(x)
        x = Dropout(self.dropout_rate)(x)

        x = Dense(100, activation=self.activation)(x)
        x = BatchNormalization()(x)

        outputs = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=['accuracy'])

    def fit(self, X, y, **kwargs):
        """
        Train the model on the provided data.

        Args:
            X (array-like): Training features
            y (array-like): Target values
            **kwargs: Additional arguments to pass to model.fit()

        Returns:
            History object: Training history
        """
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1
        )
        history = self.model.fit(
            X, y,
            validation_split=0.2,
            epochs=50,
            batch_size=1024,
            callbacks=[early_stopping],
            **kwargs
        )
        return history

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def tune(self, X, y, algorithm="RandomSearch", epochs=50, export_csv=False, **kwargs):
        """
        Perform hyperparameter tuning using specified search algorithm.

        Args:
            X (array-like): Training features
            y (array-like): Target values
            algorithm (str, optional): Search algorithm ("RandomSearch", "Hyperband", or "GridSearch").
                                     Defaults to "RandomSearch".
            epochs (int, optional): Number of training epochs. Defaults to 5.
            export_csv (bool, optional): Whether to export results to CSV. Defaults to False.
            **kwargs: Additional arguments for the tuner

        Returns:
            dict: Best hyperparameters found
        """
        features_number = X.shape[1]

        class MLPAttentionHyperModel2(HyperModel):
            """Nested HyperModel class for hyperparameter tuning."""

            def build(self, hp):
                """Build the model with hyperparameters to tune."""
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

                activation = 'relu'
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

                # Tune optimizer and learning rate
                optimizer_choice = hp.Choice('optimizer', values=['adam', 'sgd'])
                learning_rate = hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')

                if optimizer_choice == 'adam':
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                else:
                    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

                model.compile(
                    optimizer=optimizer,
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )

                return model

            def fit(self, hp, model, *args, **kwargs):
                """Define the training process with hyperparameters."""
                batch_size = hp.Choice('batch_size', values=[128, 256, 512, 1024])
                kwargs['batch_size'] = batch_size
                return model.fit(*args, **kwargs)

        # Set up tuner
        tuner_classes = {
            "RandomSearch": RandomSearch,
            "Hyperband": Hyperband,
            "GridSearch": GridSearch
        }

        if algorithm not in tuner_classes:
            raise ValueError(f"Unsupported algorithm. Choose from {', '.join(tuner_classes.keys())}")

        tuner_class = tuner_classes[algorithm]
        tuner_params = {
            "hypermodel": MLPAttentionHyperModel2(),
            "objective": 'val_accuracy',
            "directory": f"{base_path}/Results/mlp-attention2/{algorithm}/",
            "project_name": "mlp_hyperparameter_tuning_RandomSearch2024-10-30_202446",
#"project_name": f'mlp_hyperparameter_tuning_{algorithm}'
#                f'{datetime.now().strftime("%Y-%m-%d_%H%M%S")}',
            **kwargs
        }

        if algorithm == "Hyperband":
            tuner_params["max_epochs"] = epochs
        else:
            tuner_params["max_trials"] = kwargs.get('max_trials', 30)

        tuner = tuner_class(**tuner_params)

        # Perform the search
        tuner.search(X, y, epochs=epochs, validation_split=0.2)
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Export results if requested
        if export_csv:
            self._export_tuning_results(tuner, algorithm)

        # Return best parameters
        best_params = {
            "units_1": best_hp.get('units_1'),
            "units_2": best_hp.get('units_2'),
            "units_3": best_hp.get('units_3'),
            "dropout_rate": best_hp.get('dropout_rate'),
            "optimizer": best_hp.get('optimizer'),
            "learning_rate": best_hp.get('learning_rate'),
            "loss": 'binary_crossentropy',
            "num_heads": best_hp.get('num_heads'),
            "key_dim": best_hp.get('key_dim'),
            "batch_size": best_hp.get('batch_size')
        }

        return best_params

    @staticmethod
    def _export_tuning_results(self, tuner, algorithm):
        """
        Export hyperparameter tuning results to CSV files.

        Args:
            tuner: The keras-tuner instance
            algorithm (str): The search algorithm used
        """
        all_trials = tuner.oracle.trials
        trial_data = []

        for trial_id, trial in all_trials.items():
            trial_info = {
                'trial_id': trial_id,
                'score': trial.score,
                'loss': trial.metrics.get_last_value('val_loss'),
                'status': trial.status,
            }
            trial_info.update(trial.hyperparameters.values)
            trial_data.append(trial_info)

        results_df = pd.DataFrame(trial_data)
        results_df = results_df.sort_values('score', ascending=False)

        results_path = f'{base_path}/Results/mlp-attention/{algorithm}/'
        os.makedirs(results_path, exist_ok=True)

        # Save all trials
        results_df.to_csv(
            f'{results_path}/all_trials_results{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.csv',
            index=False
        )

        # Save best trial
        best_trial = results_df.iloc[0].to_dict()
        pd.DataFrame([best_trial]).to_csv(
            f'{results_path}/best_trial_results{datetime.now().strftime("%Y-%m-%d_%H%M%S")}.csv',
            index=False
        )

