from Scripts.Plots.Plotting import explain_with_shap_summary_plots, explain_with_shap_dependence_plots, \
    explain_with_force_plots
# Import the necessary libraries (tested for Python 3.9)
import os
import numpy as np

# True to print debugging outputs, False to silence the program
DEBUG = True
separator = "-------------------------------------------------------------------------"
# Define the number of clusters that will represent the
# training dataset for SHAP framework (cannot give all training samples)
K_MEANS_CLUSTERS = 100
# Define the number of testing samples on which SHAP
# will derive interpretations
SAMPLES_NUMBER = 300
# Correlation threshold for Pearson correlation.
# For feature pairs with correlation higher than the threshold,
# one feature is dropped
CORRELATION_THRESHOLD = 0.9

# Families mainly discussed within the paper
paper_families = ["bamital", "conficker", "cryptolocker", "matsnu", "suppobox", "all_DGAs"]

# Families considered for SHAP interpretations
families = ["tranco", "bamital", "banjori", "bedep", "chinad", "conficker", "corebot", "cryptolocker", "dnschanger", "dyre", "emotet", "gameover", "gozi", "locky", "matsnu", "monerominer", "murofet", "murofetweekly", "mydoom", "necurs", "nymaim2", "nymaim", "oderoor", "padcrypt", "pandabanker", "pitou", "proslikefan", "pushdo", "pykspa", "qadars", "qakbot", "qsnatch", "ramnit", "ranbyus", "rovnix", "sisron", "sphinx", "suppobox", "sutra", "symmi", "tinba", "tinynuke", "torpig", "urlzone", "vidro", "virut", "wd"]

# Dataset to load
filename = "/content/drive/MyDrive/Netmode/fedxai4dga/labeled_dataset_features.csv"


def calculate_and_explain_shap(family, algorithm, model_gs, model_explainer, test_sample, names_sample, base_path):
    print(f"Calculating SHAP values for family: {family}")

    # Create directory for results
    result_path = os.path.join(base_path, "results", algorithm, family)
    os.makedirs(result_path, exist_ok=True)

    # Calculate SHAP values
    model_shap_values = model_explainer[algorithm].shap_values(test_sample[family])
    model_shap_values = np.asarray(model_shap_values)
    model_shap_values = model_shap_values[:, :, 0]

    # Generate explanations
    explain_with_shap_summary_plots(model_gs[algorithm], model_shap_values, family, test_sample[family], algorithm)

    explain_with_shap_dependence_plots(
        model_gs[algorithm],
        model_shap_values,
        family,
        test_sample[family],
        "Reputation", "Length", "Words_Mean", "Max_Let_Seq", "Words_Freq",
        "Vowel_Freq", "Entropy", "DeciDig_Freq", "Max_DeciDig_Seq",
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
