import tensorflow as tf
from sklearn.metrics import accuracy_score
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model

from .ModelBase import ModelBase


class MLPAttentionModel2(ModelBase):
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
        x_reshaped = tf.expand_dims(inputs, axis=1)
        attention_output = MultiHeadAttention(num_heads=8, key_dim=50)(x_reshaped, x_reshaped)

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
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=['accuracy'])
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


"""
# Define the self-attention layer
class SelfAttentionLayer(layers.Layer):
    def __init__(self, dropout_rate=0.2, activation='relu',
                 optimizer='adam', loss='binary_crossentropy'):
        super(SelfAttentionLayer, self).__init__()
        self.W_Q = layers.Dense(output_dim)
        self.W_K = layers.Dense(output_dim)
        self.W_V = layers.Dense(output_dim)
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.optimizer = optimizer
        self.loss = loss
        self.model = None

    def call(self, inputs):
        Q = self.W_Q(inputs)
        K = self.W_K(inputs)
        V = self.W_V(inputs)

        # Calculate attention scores
        scores = tf.matmul(Q, K, transpose_b=True)
        scores = scores / tf.sqrt(tf.cast(tf.shape(K)[-1], tf.float32))
        weights = tf.nn.softmax(scores)

        # Weighted sum of values
        output = tf.matmul(weights, V)
        return output

    # Build the MLP model with self-attention
    def build(self, input_shape):
        inputs = layers.Input(shape=input_shape)
        attention_output = SelfAttentionLayer(64)(inputs)
        x = layers.Flatten()(attention_output)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        outputs = layers.Dense(1, activation='softmax')(x)  # For multi-class

        model = models.Model(inputs, outputs)
        return model

    def fit(self, X, y, **kwargs):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        self.model.fit(X, y, validation_split=0.2, epochs=10, batch_size=512,
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

"""