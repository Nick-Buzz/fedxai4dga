from Models.xgboost_model import XGBoostModel
from Models.mlp_model import MLPModel
from Models.mlp_attention_model import MLPAttentionModel


def train_model(X_train, y_train, algorithm):
    models = {
        "xgboost": XGBoostModel,
        "mlp": MLPModel,
        "mlp-attention": MLPAttentionModel
    }

    if algorithm not in models:
        raise ValueError(f"Algorithm '{algorithm}' not supported")

    model = models[algorithm]()

    # For MLPAttentionModel, we need to pass features_number
    if algorithm == "mlp-attention":
        model.build(features_number=X_train.shape[1])
    else:
        model.build()

    # If you want to specify parameters on training go to fit function
    # on each model
    model.fit(X_train, y_train)

    # For XGBoost, the model is directly stored in the class
    # For TensorFlow models, it's stored in the 'model' attribute
    return model.model if hasattr(model, 'model') else model

# Usage example
# trained_model = train_model(X_train, y_train, "mlp-attention")