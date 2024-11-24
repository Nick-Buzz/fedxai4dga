from Models.xgboost_model import XGBoostModel
from Models.mlp_model import MLPModel
from Models.mlp_attention_model import MLPAttentionModel
from Models.mlp_attention_model2 import MLPAttentionModel2
from Models.AutoEncoder import AutoencoderModel
from Scripts.Plots.Plotting import plot_training_curves


def train_model(X_train, y_train, algorithm, tune,save_path):
    models = {
        "xgboost": XGBoostModel,
        "mlp": MLPModel,
        "mlp-attention": MLPAttentionModel,
        "mlp-attention2": MLPAttentionModel2,
        "autoencoder": AutoencoderModel
    }

    if algorithm not in models:
        raise ValueError(f"Algorithm '{algorithm}' not supported")

    model = models[algorithm]()

    # For MLPAttentionModel, we need to pass features_number
    if "xgboost" not in algorithm:
        if tune:
            best_params = model.tune(X_train, y_train)
            model.set_params(**best_params).build(features_number=X_train.shape[1])

        model.build(features_number=X_train.shape[1])
    else:
        model.build()

    # If you want to specify parameters on training go to fit function
    # on each model
    history = model.fit(X_train, y_train)

    plot_training_curves(algorithm, history,save_path=save_path)

    # For XGBoost, the model is directly stored in the class
    # For TensorFlow models, it's stored in the 'model' attribute
    return model.model if hasattr(model, 'model') else model

# Usage example
# trained_model = train_model(X_train, y_train, "mlp-attention")