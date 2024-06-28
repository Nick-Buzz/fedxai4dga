import numpy as np


def evaluate_model_on_family(model, family, x_testing, algorithm):
    # Calculate accuracy for a specific malware family
    predictions = model.predict(x_testing)
    non_zero_values = np.count_nonzero(predictions)
    if family != "tranco":
        accuracy = non_zero_values / predictions.size
    else:
        zero_values = predictions.size - non_zero_values
        accuracy = zero_values / predictions.size
    print("Accuracy on sampled testing dataset for family and algorithm: ", family, algorithm, accuracy)
    return None
