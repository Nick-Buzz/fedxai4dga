import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, Concatenate
from tensorflow.keras.models import Model
from .ModelBase import ModelBase


class MLPAttentionModel(ModelBase):
    def __init__(self, dropout_rate=0.2, activation='relu',
                 optimizer='adam', loss='binary_crossentropy'):
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.model = None

    def build(self, features_number):
        # Define the inputs
        inputs = Input(shape=(features_number,))

        # Define the first MLP module
        x = Dense(300, activation=self.activation)(inputs)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(200, activation=self.activation)(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(200, activation='linear')(x)
        x = Dropout(self.dropout_rate)(x)

        # Reshape for the MultiHeadAttention layer
        x_reshaped = tf.expand_dims(x, axis=1)

        # Define the self-attention layer
        attention_output = MultiHeadAttention(num_heads=8, key_dim=200)(x_reshaped, x_reshaped)

        # Remove the added dimension
        attention_output = tf.squeeze(attention_output, axis=1)

        # Concatenate the attention output with the MLP output
        combined = Concatenate()([x, attention_output])

        # Define the second MLP module
        x = Dense(300, activation=self.activation)(combined)
        x = Dense(200, activation=self.activation)(x)
        outputs = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
        # return self.model

    def fit(self, X, y, **kwargs):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        self.model.fit(X, y, validation_split=0.2, epochs=100, batch_size=512,
                       callbacks=[early_stopping], **kwargs)

    def score(self, X, y, sample_weight=None):

        return accuracy_score(y, self.model.predict(X), sample_weight=sample_weight)

    def get_params(self, deep=True):
        return {
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

