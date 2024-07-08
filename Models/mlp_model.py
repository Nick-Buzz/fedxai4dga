import tensorflow as tf
from .ModelBase import ModelBase


class MLPModel(ModelBase):
    def __init__(self):
        self.model = None

    def build(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(300, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    def fit(self, X_train, y_train):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1)
        self.model.fit(X_train, y_train, validation_split=0.2, epochs=10, batch_size=512, callbacks=[early_stopping])