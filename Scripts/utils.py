from Scripts.Plots.Plotting import explain_with_shap_summary_plots, explain_with_shap_dependence_plots, \
    explain_with_force_plots
# Import the necessary libraries (tested for Python 3.9)
import os
import numpy as np


def calculate_and_explain_shap(family, algorithm, model_gs, model_explainer, test_sample, names_sample, base_path):
    print(f"Calculating SHAP values for family: {family}")

    # Create directory for results
    result_path = os.path.join(base_path, "Results", algorithm, family)
    os.makedirs(result_path, exist_ok=True)

    # Calculate SHAP values
    model_shap_values = model_explainer[algorithm].shap_values(test_sample[family])
    model_shap_values = np.asarray(model_shap_values)
    model_shap_values = model_shap_values[:, :, 0]

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
