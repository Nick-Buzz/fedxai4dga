from Scripts.Plots.Plotting import explain_with_shap_summary_plots, explain_with_shap_dependence_plots, \
    explain_with_force_plots
# Import the necessary libraries (tested for Python 3.9)
import os
import numpy as np
import tensorflow as tf


def calculate_and_explain_shap(family, algorithm, model_gs, model_explainer, test_sample, names_sample, base_path):
    print(f"Calculating SHAP values for family: {family}")

    # Create directory for results
    result_path = os.path.join(base_path, "Results", algorithm, family)
    os.makedirs(result_path, exist_ok=True)

    # Calculate SHAP values
    model_shap_values = model_explainer[algorithm].shap_values(test_sample[family])
    model_shap_values = np.asarray(model_shap_values)
    model_shap_values = model_shap_values[0]
    print(model_shap_values.shape)
    print(test_sample[family].shape)

    # Generate explanations
    explain_with_shap_summary_plots(model_gs[algorithm],
                                    model_shap_values,
                                    family,
                                    test_sample[family],
                                    algorithm)

    explain_with_shap_dependence_plots(
        model_gs[algorithm],
        model_shap_values,
        family,
        test_sample[family],
        "Reputation",  "Length",       "Words_Mean",
        "Max_Let_Seq", "Words_Freq",   "Vowel_Freq",
        "Entropy",     "DeciDig_Freq", "Max_DeciDig_Seq",
        algorithm
    )

    explain_with_force_plots(
        model_gs[algorithm],
        model_shap_values,
        family,
        test_sample[family],
        names_sample[family],
        algorithm,
        model_explainer[algorithm]
    )


# Usage example:
# calculate_and_explain_shap(
#     family="bamital",
#     algorithm="mlp",
#     model_gs=model_gs,
#     model_explainer=model_explainer,
#     test_sample=test_sample,
#     names_sample=names_sample,
#     base_path="/content/drive/MyDrive/Netmode/fedxai4dga"
# )


def get_mlp_attention_info(model):
    """Extract detailed information about MLP and attention layers."""
    model_info = {
        'mlp1_layers': [],
        'attention_layer': None,
        'mlp2_layers': [],
        'training_params': {},
        'architecture_params': {}
    }

    in_mlp2 = False
    for layer in model.layers:
        config = layer.get_config()

        if isinstance(layer, tf.keras.layers.Dense):
            layer_info = {
                'units': layer.units,
                'activation': config['activation'],
                'trainable_params': layer.count_params(),
                'output_shape': layer.output_shape
            }
            if not in_mlp2:
                model_info['mlp1_layers'].append(layer_info)
            else:
                model_info['mlp2_layers'].append(layer_info)

        elif isinstance(layer, tf.keras.layers.MultiHeadAttention):
            model_info['attention_layer'] = {
                'num_heads': layer.num_heads,
                'key_dim': layer.key_dim,
                'value_dim': layer.value_dim if hasattr(layer, 'value_dim') else None,
                'dropout': layer.dropout if hasattr(layer, 'dropout') else None,
                'output_shape': layer.output_shape,
                'trainable_params': layer.count_params()
            }
            in_mlp2 = True

        elif isinstance(layer, tf.keras.layers.Dropout):
            model_info['architecture_params']['dropout_rate'] = config['rate']

    # Get training parameters
    if hasattr(model, 'optimizer'):
        model_info['training_params']['optimizer'] = {
            'name': model.optimizer.__class__.__name__,
            'config': model.optimizer.get_config()
        }

    if hasattr(model, 'loss'):
        model_info['training_params']['loss'] = model.loss if isinstance(model.loss,
                                                                         str) else model.loss.__class__.__name__

    return model_info


def get_model_architecture(model, algorithm):
    """Extract model architecture details with support for attention layers."""
    try:

        # Handle Keras-based models (MLP, CNN, Transformer, etc) if `get_params` is not available
        if hasattr(model, 'layers'):
            string_list = []
            model.summary(print_fn=lambda x: string_list.append(x))

            config = []
            for layer in model.layers:
                # Basic layer info
                layer_config = {
                    'name': layer.name,
                    'type': layer.__class__.__name__,
                    'units': getattr(layer, 'units', None),
                    'activation': getattr(layer.activation, '__name__', str(layer.activation)) if hasattr(layer,
                                                                                                          'activation') else None
                }

                # Add attention-specific attributes
                if 'Attention' in layer.__class__.__name__:
                    layer_config['num_heads'] = getattr(layer, 'num_heads', 'Not available')
                    layer_config['key_dim'] = getattr(layer, 'key_dim', 'Not available')

                config.append(layer_config)

            return {
                'summary': '\n'.join(string_list),
                'config': config,
                'total_params': model.count_params()
            }

        # Handle other model types like XGBoost
        elif algorithm == "xgboost":
            return {
                'booster': str(model.get_booster()),
                'params': model.get_params()
            }

        else:
            return "Unsupported model type"

    except Exception as e:
        return f"Error extracting architecture: {str(e)}"


def get_model_architecture1(model, algorithm):
    """Extract model architecture details with support for attention layers."""
    try:
        # Handle Keras-based models (MLP, CNN, Transformer, etc)
        if hasattr(model, 'layers'):
            string_list = []
            model.summary(print_fn=lambda x: string_list.append(x))

            config = []
            for layer in model.layers:
                # Basic layer info
                layer_config = {
                    'name': layer.name,
                    'type': layer.__class__.__name__
                }

                # Add common layer attributes if they exist
                if hasattr(layer, 'units'):
                    layer_config['units'] = layer.units
                if hasattr(layer, 'activation'):
                    layer_config['activation'] = layer.activation.__name__ if hasattr(layer.activation,
                                                                                      '__name__') else str(
                        layer.activation)

                # Add attention-specific attributes
                if 'attention' in layer.__class__.__name__:
                    if hasattr(layer, 'num_heads'):
                        layer_config['num_heads'] = layer.num_heads
                    if hasattr(layer, 'key_dim'):
                        layer_config['key_dim'] = layer.key_dim

                config.append(layer_config)

            return {
                'summary': '\n'.join(string_list),
                'config': config,
                'total_params': model.count_params()
            }

        # Handle XGBoost
        elif algorithm == "xgboost":
            return {
                'booster': str(model.get_booster()),
                'params': model.get_params()
            }

        else:
            return "Unsupported model type"

    except Exception as e:
        return f"Error extracting architecture: {str(e)}"