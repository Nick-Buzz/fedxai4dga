"""ExAt-MLP: attention-enhanced MLP for tabular DGA detection.

Given N scalar features x = [x_1, ..., x_N]:

  1. Feature projection.  Each x_i is mapped independently to an embedding
     e_i = x_i * W_i + b_i in R^{d_emb} (FeatureTokenizer), giving a sequence
     E in R^{N x d_emb} with one token per feature.
  2. Multi-head self-attention.  E is projected to Query, Key and Value in each
     of H heads (key dimension d_k); the heads are concatenated and linearly
     projected back to A in R^{N x d_emb}.  A single block, applied without
     residual or normalization sub-layers; A is used directly.
  3. Readout.  A is flattened to N * d_emb values and concatenated with the raw
     input x (N * d_emb + N values), which feeds the MLP classifier.
  4. MLP classifier.  Dense layers with dropout, then a sigmoid output unit.

Defaults: d_emb = 60, H = 12, d_k = 64, dense 400/200/100, dropout 0.3,
Adam learning rate 1e-3.  With N = 49 the concatenated vector has 2,989 values
and the model about 1.5M trainable parameters.

Four configurations share this one class, so every ablation differs from the
proposed model in exactly one component:

    ExAtMLP()                       proposed model
    ExAtMLP(use_attention=False)    identical, attention block removed
    ExAtMLP(num_heads=1)            single-head ablation
    ExAtMLP(use_tokens=False)       plain dense stack on the raw features
"""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


@tf.keras.utils.register_keras_serializable(package="exat_mlp")
class FeatureTokenizer(layers.Layer):
    """Embed each scalar feature as its own token.

    Input   (batch, n_features)
    Output  (batch, n_features, d_model)

    ``e_i = x_i * W_i + b_i`` with a distinct W_i and b_i per feature.  The
    per-feature parameters are what give the tokens identity; a single shared
    Dense layer would map every feature carrying the same value to the same
    vector and attention would have no basis for distinguishing them.
    """

    def __init__(self, d_model: int = 64, **kwargs):
        super().__init__(**kwargs)
        self.d_model = int(d_model)

    def build(self, input_shape):
        n_features = int(input_shape[-1])
        self.n_features = n_features
        self.kernel = self.add_weight(
            name="kernel",
            shape=(n_features, self.d_model),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.bias = self.add_weight(
            name="bias",
            shape=(n_features, self.d_model),
            initializer="zeros",
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        # (B, N) -> (B, N, 1) -> broadcast against (N, d) -> (B, N, d)
        return tf.expand_dims(inputs, axis=-1) * self.kernel + self.bias

    def compute_output_shape(self, input_shape):
        return tuple(input_shape) + (self.d_model,)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


class ExAtMLP:
    """Attention-enhanced MLP.

    ``Scripts/run_experiments.py`` drives every variant through the same
    interface: ``build(features_number=...)``, ``fit(X, y, validation_data=...,
    ...)`` and a ``.model`` attribute holding the compiled Keras model.
    """

    def __init__(
        self,
        d_model: int = 60,
        num_heads: int = 12,
        key_dim: int = 64,
        attention_dropout: float = 0.0,
        hidden_layers=(400, 200, 100),
        dropout_rate: float = 0.3,
        activation: str = "relu",
        learning_rate: float = 1e-3,
        loss: str = "binary_crossentropy",
        use_attention: bool = True,
        use_tokens: bool = True,
        name: str = "exat_mlp",
    ):
        self.d_model = int(d_model)          # d_emb, the per-feature embedding size
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        self.attention_dropout = float(attention_dropout)
        self.hidden_layers = tuple(int(u) for u in hidden_layers)
        self.dropout_rate = float(dropout_rate)
        self.activation = activation
        self.learning_rate = float(learning_rate)
        self.loss = loss
        self.use_attention = bool(use_attention)
        self.use_tokens = bool(use_tokens)
        self.name = name
        self.model = None

    # ------------------------------------------------------------------ build

    def build(self, features_number: int):
        inputs = layers.Input(shape=(features_number,), name="features")

        if self.use_tokens:
            branch = self._token_branch(inputs)
            combined = layers.Concatenate(name="concat_raw_and_attention")([inputs, branch])
        else:
            combined = inputs

        x = combined
        for i, units in enumerate(self.hidden_layers, start=1):
            x = layers.Dense(units, activation=self.activation, name=f"dense_{i}")(x)
            x = layers.Dropout(self.dropout_rate, name=f"drop_{i}")(x)

        outputs = layers.Dense(1, activation="sigmoid", name="prediction")(x)

        self.model = Model(inputs=inputs, outputs=outputs, name=self.name)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.loss,
            metrics=["accuracy"],
        )
        return self.model

    def _token_branch(self, inputs):
        """Project -> (multi-head self-attention) -> flatten.  Returns (B, N * d_model)."""
        tokens = FeatureTokenizer(self.d_model, name="feature_tokenizer")(inputs)

        if self.use_attention:
            # Single block, no residual or normalization: A is used directly.
            # The output projection maps the concatenated heads back to d_model.
            tokens = layers.MultiHeadAttention(
                num_heads=self.num_heads,
                key_dim=self.key_dim,
                dropout=self.attention_dropout,
                name="multi_head_attention",
            )(tokens, tokens)

        return layers.Flatten(name="flatten_attention")(tokens)

    # -------------------------------------------------------------------- fit

    def fit(
        self,
        X,
        y,
        validation_data=None,
        epochs: int = 50,
        batch_size: int = 1024,
        patience: int = 10,
        callbacks=None,
        **kwargs,
    ):
        """Train with an explicit validation set.

        ``validation_data`` is required rather than optional.  Keras'
        ``validation_split`` takes the final fraction of the arrays before
        shuffling, and imbalanced-learn's SMOTE appends its synthetic rows to
        the end, so ``validation_split`` after SMOTE would validate mostly on
        synthetic rows.  Hold the validation set out before SMOTE instead.
        """
        if self.model is None:
            raise RuntimeError("call build(features_number=...) before fit()")
        if validation_data is None:
            raise ValueError(
                "validation_data is required. Hold the validation set out of the "
                "original training rows with stratification BEFORE applying SMOTE, "
                "and fit SMOTE only on the remaining training subset."
            )

        fit_callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=patience,
                restore_best_weights=True,
                verbose=1,
            )
        ]
        if callbacks:
            fit_callbacks.extend(callbacks)

        return self.model.fit(
            X,
            y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=fit_callbacks,
            **kwargs,
        )

    # ---------------------------------------------------------------- helpers

    def predict_proba(self, X, **kwargs):
        return self.model.predict(X, **kwargs).reshape(-1)

    def get_params(self, deep=True):
        return {
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "attention_dropout": self.attention_dropout,
            "hidden_layers": self.hidden_layers,
            "dropout_rate": self.dropout_rate,
            "activation": self.activation,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "use_attention": self.use_attention,
            "use_tokens": self.use_tokens,
        }

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self


# Named configurations used by the experiment harness.  Every ablation differs
# from VARIANTS["exat_mlp"] in exactly one argument.
VARIANTS = {
    "exat_mlp":          dict(),
    "no_attention":      dict(use_attention=False, name="no_attention"),
    "single_head":       dict(num_heads=1, name="single_head"),
    "plain_mlp":         dict(use_tokens=False, name="plain_mlp"),
}


def build_variant(variant: str, features_number: int, **overrides) -> ExAtMLP:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {sorted(VARIANTS)}")
    kwargs = dict(VARIANTS[variant])
    kwargs.update(overrides)
    wrapper = ExAtMLP(**kwargs)
    wrapper.build(features_number=features_number)
    return wrapper
