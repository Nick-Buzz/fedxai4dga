# Import the necessary libraries (tested for Python 3.9)
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.python.keras import Sequential
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
from pdpbox import pdp, info_plots
import shap

# True to print debugging outputs, False to silence the program
DEBUG = True
separator = "-------------------------------------------------------------------------"
# Define the number of clusters that will represent the training dataset
# for SHAP framework (cannot give all training samples)
K_MEANS_CLUSTERS = 100
# Define the number of testing samples on which SHAP will derive interpretations
SAMPLES_NUMBER = 300
# Correlation threshold for Pearson correlation. For feature pairs with
# correlation higher than the threshold, one feature is dropped
CORRELATION_THRESHOLD = 0.9

# Families mainly discussed within the paper
paper_families = ["bamital", "conficker", "cryptolocker", "matsnu", "suppobox", "all_DGAs"]

# Families considered for SHAP interpretations
families = ["tranco", "bamital", "banjori", "bedep", "chinad", "conficker", "corebot", "cryptolocker", "dnschanger", "dyre", "emotet", "gameover", "gozi", "locky", "matsnu", "monerominer", "murofet", "murofetweekly", "mydoom", "necurs", "nymaim2", "nymaim", "oderoor", "padcrypt", "pandabanker", "pitou", "proslikefan", "pushdo", "pykspa", "qadars", "qakbot", "qsnatch", "ramnit", "ranbyus", "rovnix", "sisron", "sphinx", "suppobox", "sutra", "symmi", "tinba", "tinynuke", "torpig", "urlzone", "vidro", "virut", "wd"]


def plot_1d_pdp(model, family, family_data, feature_to_plot, features, algorithm):
    # Plot Partial Dependency Plots in one dimension (No SHAP)
    feature_to_plot = str(feature_to_plot)
    pdp_dist = pdp.pdp_isolate(model=model, dataset= family_data, model_features = features, feature = feature_to_plot)
    pdp.pdp_plot(pdp_dist, feature_to_plot)
    name = "./results/pdp-" + str(family) + "-" + str(feature_to_plot) + "-original" + str(algorithm) + ".png"
    plt.savefig(name)
    plt.close("all")

    plt.xlim(0, 1)
    plt.ylim(-1, 1)
    name = "./results/pdp-" + str(family) + "-" + str(feature_to_plot) + "-xlim01-ylim-11" + str(algorithm) + ".png"
    plt.savefig(name)
    plt.close("all")

    plt.xlim(0, 1)
    plt.ylim(-1, 0)
    name = "./results/pdp-" + str(family) + "-" + str(feature_to_plot) + "-xlim01-ylim-10" + str(algorithm) + ".png"
    plt.savefig(name)
    plt.close("all")
    return None


def plot_2d_pdp(model, family, family_data, feature1, feature2, features, algorithm):
    # Plot 2D Partial Dependency Plots (No SHAP)
    features_to_plot = [str(feature1), str(feature2)]
    inter1 = pdp.pdp_interact(model = model, dataset=family_data, model_features= features, features = features_to_plot)
    pdp.pdp_interact_plot (pdp_interact_out=inter1, feature_names=features_to_plot)
    name = "./results/2dpdp-" + str(family) + "-" + str(feature1) + "-" + str(feature2) + "-" + str(algorithm) + ".png"
    plt.savefig(name)
    plt.close("all")
    return None


def explain_with_shap_summary_plots(model, model_shap_values, family, test_sample, algorithm):
    # Plot bar summary plot using SHAP values
    prepend_path = "/content/drive/MyDrive/Netmode/fedxai4dga/results/" + str(algorithm) + "/" + str(family) + "/summary-plots/"
    os.makedirs(prepend_path, exist_ok=True)

    fig = plt.clf()
    shap.summary_plot(model_shap_values, test_sample, plot_type = "bar", show = False)
    name = prepend_path + str(family) + "-summarybar-original-" + str(algorithm) + ".png"
    plt.savefig(name)
    plt.close("all")

    plt.xlim(-1, 1)
    name = prepend_path + str(family) + "-summarybar-xlim-11-" + str(algorithm) + ".png"
    plt.savefig(name)
    plt.close("all")

    # Plot summary plot using SHAP values
    fig = plt.clf()
    shap.summary_plot(model_shap_values, test_sample, show = False)
    name = prepend_path + str(family) + "-summarynotbar-original-" + str(algorithm) + ".png"
    plt.savefig(name)
    plt.close("all")

    plt.xlim(-1, 1)
    name = prepend_path + str(family) + "-summarynotbar-xlim-11-" + str(algorithm) + ".png"
    plt.savefig(name)
    plt.close("all")


def explain_with_shap_dependence_plots(model, model_shap_values, family, test_sample, feature1, feature2, feature3,
                                       feature4, feature5, feature6, feature7, feature8, feature9, algorithm):
    # Plot dependence plot using SHAP values for multiple features
    prepend_path = "/content/drive/MyDrive/Netmode/fedxai4dga/results/" + str(algorithm) + "/" + str(
        family) + "/dependence-plots/"
    os.makedirs(prepend_path, exist_ok=True)

    fig = plt.clf()
    shap.dependence_plot(feature1, model_shap_values, test_sample, show=False)
    name = prepend_path + str(family) + "-dependence-" + str(feature1) + "-original-" + str(algorithm) + ".png"
    plt.savefig(name, bbox_inches='tight')
    plt.close("all")

    fig = plt.clf()
    shap.dependence_plot(str(feature2), model_shap_values, test_sample, show=False)
    name = prepend_path + str(family) + "-dependence-" + str(feature2) + "-original-" + str(algorithm) + ".png"
    plt.savefig(name, bbox_inches='tight')
    plt.close("all")

    fig = plt.clf()
    shap.dependence_plot(str(feature3), model_shap_values, test_sample, show=False)
    name = prepend_path + str(family) + "-dependence-" + str(feature3) + "-original-" + str(algorithm) + ".png"
    plt.savefig(name, bbox_inches='tight')
    plt.close("all")

    fig = plt.clf()
    shap.dependence_plot(str(feature4), model_shap_values, test_sample, show=False)
    name = prepend_path + str(family) + "-dependence-" + str(feature4) + "-original-" + str(algorithm) + ".png"
    plt.savefig(name, bbox_inches='tight')
    plt.close("all")

    fig = plt.clf()
    shap.dependence_plot(str(feature5), model_shap_values, test_sample, show=False)
    name = prepend_path + str(family) + "-dependence-" + str(feature5) + "-original-" + str(algorithm) + ".png"
    plt.savefig(name, bbox_inches='tight')
    plt.close("all")

    fig = plt.clf()
    shap.dependence_plot(str(feature6), model_shap_values, test_sample, show=False)
    name = prepend_path + str(family) + "-dependence-" + str(feature6) + "-original-" + str(algorithm) + ".png"
    plt.savefig(name, bbox_inches='tight')
    plt.close("all")

    fig = plt.clf()
    shap.dependence_plot(str(feature7), model_shap_values, test_sample, show=False)
    name = prepend_path + str(family) + "-dependence-" + str(feature7) + "-original-" + str(algorithm) + ".png"
    plt.savefig(name, bbox_inches='tight')
    plt.close("all")

    fig = plt.clf()
    shap.dependence_plot(str(feature8), model_shap_values, test_sample, show=False)
    name = prepend_path + str(family) + "-dependence-" + str(feature8) + "-original-" + str(algorithm) + ".png"
    plt.savefig(name, bbox_inches='tight')
    plt.close("all")

    fig = plt.clf()
    shap.dependence_plot(str(feature9), model_shap_values, test_sample, show=False)
    name = prepend_path + str(family) + "-dependence-" + str(feature9) + "-original-" + str(algorithm) + ".png"
    plt.savefig(name, bbox_inches='tight')
    plt.close("all")

    return None


def explain_with_force_plots(model, model_shap_values, family, test_sample, names_sample, algorithm,
                             model_explainer):
    # Plot force plots using SHAP values (local explanations)
    prepend_path = "/content/drive/MyDrive/Netmode/fedxai4dga/results/" + str(algorithm) + "/" + str(
        family) + "/force-plots/"
    os.makedirs(prepend_path, exist_ok=True)

    predictions = model.predict(test_sample)
    index_values = list(test_sample.index.values)
    sequence = 0
    for index in index_values:
        original_name = names_sample[index]
        name = original_name.replace(".", "+")
        prediction = predictions[sequence]

        fig = plt.clf()
        shap.force_plot(model_explainer.expected_value, model_shap_values[sequence, :], test_sample.loc[index],
                        matplotlib=True, show=False)
        name_of_file = prepend_path + str(family) + "-force-" + str(sequence) + "-name-" + str(
            name) + "-prediction-" + str(prediction) + "-" + str(algorithm) + "-original.png"
        plt.title(original_name, y=1.5)
        plt.savefig(name_of_file, bbox_inches='tight')
        plt.close("all")

        fig = plt.clf()
        shap.force_plot(model_explainer.expected_value, model_shap_values[sequence, :], test_sample.loc[index],
                        matplotlib=True, show=False, contribution_threshold=0.1)
        name_of_file = prepend_path + str(family) + "-force-" + str(sequence) + "-name-" + str(
            name) + "-prediction-" + str(prediction) + "-" + str(algorithm) + "-threshold01.png"
        plt.title(original_name, y=1.5)
        plt.savefig(name_of_file, bbox_inches='tight')
        plt.close("all")

        sequence += 1
        # Plot only the first 100 or less if no more than 100 exist
        if sequence == 1000:
            break
    return None

