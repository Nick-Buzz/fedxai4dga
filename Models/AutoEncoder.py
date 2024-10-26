import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error, accuracy_score
from .ModelBase import ModelBase


class AutoencoderModel(ModelBase):
    def __init__(self, hidden_layer_sizes=[75,100,50,20], latent_dim=10, batch_norm=False,
                 dropout=True, dropout_rate=0.1, activation='relu',
                 optimizer='adam', threshold=0.2):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.threshold =threshold
        self.latent_dim = latent_dim
        self.batch_norm = batch_norm
        self.dropout = dropout
        self.dropout_rate = dropout_rate
        self.activation = activation
        self.optimizer = optimizer
        self.loss = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)
        self.model = None
        self.encoder = None
        self.decoder = None

    def build(self, features_number):
        # Build Encoder
        encoder_inputs = Input(shape=(features_number,))
        x = encoder_inputs
        for i, size in enumerate(self.hidden_layer_sizes):
            x = Dense(size, name=f"Encoder-Layer{i + 1}")(x)
            if self.batch_norm:
                x = BatchNormalization(name=f'Encoder-Layer-Normalization{i + 1}')(x)
            x = LeakyReLU(name=f'Encoder-Layer-Activation{i + 1}')(x)
            if self.dropout:
                x = Dropout(rate=self.dropout_rate, name=f"Encoder-Layer{i + 1}-Dropout")(x)
        latent_space = Dense(self.latent_dim, name="Latent_space-Layer")(x)
        self.encoder = Model(encoder_inputs, latent_space, name="encoder")

        # Build Decoder
        decoder_inputs = Input(shape=(self.latent_dim,))
        x = decoder_inputs
        for i, size in enumerate(reversed(self.hidden_layer_sizes)):
            x = Dense(size, name=f"Decoder-Layer{i + 1}")(x)
            if self.batch_norm:
                x = BatchNormalization(name=f'Decoder-Layer-Normalization{i + 1}')(x)
            x = LeakyReLU(name=f'Decoder-Layer-Activation{i + 1}')(x)
            if self.dropout:
                x = Dropout(rate=self.dropout_rate, name=f"Decoder-Layer{i + 1}-Dropout")(x)
        decoder_outputs = Dense(features_number, name="Output-Layer")(x)
        self.decoder = Model(decoder_inputs, decoder_outputs, name="decoder")

        # Build full autoencoder
        inputs = Input(shape=(features_number,))
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        self.model = Model(inputs, decoded)

        self.model.compile(optimizer=self.optimizer, loss=self.loss)

    def fit(self, X, y=None, **kwargs):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        history = self.model.fit(X, X, validation_split=0.2, epochs=20, batch_size=512,
                                 callbacks=[early_stopping], **kwargs)
        return history

    def predict(self, X):
        reconstructed = self.model.predict(X)
        predicted = (abs(self.loss(X, reconstructed)) > self.threshold).astype(int).flatten()
        return predicted

    def get_params(self, deep=True):
        return {
            "hidden_layer_sizes": self.hidden_layer_sizes,
            "latent_dim": self.latent_dim,
            "batch_norm": self.batch_norm,
            "dropout": self.dropout,
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
        if self.encoder is not None:
            state['encoder_config'] = self.encoder.get_config()
            state['encoder_weights'] = self.encoder.get_weights()
            del state['encoder']
        if self.decoder is not None:
            state['decoder_config'] = self.decoder.get_config()
            state['decoder_weights'] = self.decoder.get_weights()
            del state['decoder']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'model_config' in state:
            self.model = tf.keras.models.model_from_config(state['model_config'])
            self.model.set_weights(state['model_weights'])
        if 'encoder_config' in state:
            self.encoder = tf.keras.models.model_from_config(state['encoder_config'])
            self.encoder.set_weights(state['encoder_weights'])
        if 'decoder_config' in state:
            self.decoder = tf.keras.models.model_from_config(state['decoder_config'])
            self.decoder.set_weights(state['decoder_weights'])