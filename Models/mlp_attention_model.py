import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, MultiHeadAttention, Concatenate
from tensorflow.keras.models import Model
from model_base import ModelBase


class MLPAttentionModel(ModelBase):
    def __init__(self):
        self.model = None

    def build(self, features_number):
        # Define the inputs
        inputs = Input(shape=(features_number,))

        # Define the first MLP module
        x = Dense(300, activation='relu')(inputs)
        x = Dropout(0.2)(x)
        x = Dense(200, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(200, activation='linear')(x)
        x = Dropout(0.2)(x)

        # Reshape for the MultiHeadAttention layer
        x_reshaped = tf.expand_dims(x, axis=1)

        # Define the self-attention layer
        attention_output = MultiHeadAttention(num_heads=8, key_dim=200)(x_reshaped, x_reshaped)

        # Remove the added dimension
        attention_output = tf.squeeze(attention_output, axis=1)

        # Concatenate the attention output with the MLP output
        combined = Concatenate()([x, attention_output])

        # Define the second MLP module
        x = Dense(300, activation='relu')(combined)
        x = Dense(200, activation='relu')(x)
        outputs = Dense(1, activation='sigmoid')(x)

        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return self.model

    def fit(self, X_train, y_train):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
        self.model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=512, callbacks=[early_stopping])