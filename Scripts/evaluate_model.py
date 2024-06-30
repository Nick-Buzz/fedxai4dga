# True to print debugging outputs, False to silence the program
DEBUG = True
separator = "-------------------------------------------------------------------------"
# Define the number of clusters that will represent the training dataset for SHAP framework (cannot give all training samples)
K_MEANS_CLUSTERS = 100
# Define the number of testing samples on which SHAP will derive interpretations
SAMPLES_NUMBER = 300
# Correlation threshold for Pearson correlation. For feature pairs with correlation higher than the threshold, one feature is dropped
CORRELATION_THRESHOLD = 0.9

# Import the necessary libraries (tested for Python 3.8)
import os
import numpy as np
import pandas as pd
from sklearn import ensemble
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf


# Families mainly discussed within the paper
paper_families = ["bamital", "conficker", "cryptolocker", "matsnu", "suppobox", "all_DGAs"]

# Families considered for SHAP interpretations
families = ["tranco", "bamital", "banjori", "bedep", "chinad", "conficker", "corebot", "cryptolocker", "dnschanger", "dyre", "emotet", "gameover", "gozi", "locky", "matsnu", "monerominer", "murofet", "murofetweekly", "mydoom", "necurs", "nymaim2", "nymaim", "oderoor", "padcrypt", "pandabanker", "pitou", "proslikefan", "pushdo", "pykspa", "qadars", "qakbot", "qsnatch", "ramnit", "ranbyus", "rovnix", "sisron", "sphinx", "suppobox", "sutra", "symmi", "tinba", "tinynuke", "torpig", "urlzone", "vidro", "virut", "wd"]


def evaluate_model(model, X_test, y_test, algorithm):
    # Make predictions on the testing dataset
    if algorithm == "xgboost":
        predictions = model.predict(X_test)

        # Print the different testing scores
        print("Algorithm: ", str(algorithm))
        print("Accuracy: ", accuracy_score(y_test["Label"].values, predictions, normalize = True))
        print("Precision None: ", precision_score(y_test["Label"].values, predictions, average = None))
        print("Recall None: ", recall_score(y_test["Label"].values, predictions, average = None))
        print("F1 score None: ", f1_score(y_test["Label"].values, predictions, average = None))
        print(separator)
    elif algorithm == "mlp":
        # Print a summary of the MLP architecture
        print(model.summary())
        print(separator)
        # We need only the binary labels, not the domain name and the malware family
        y_test_temp = y_test.iloc[:, 1]
        score = model.evaluate(X_test, y_test_temp, verbose=1)
        print(score[0])
        print(score[1])
        print(separator)
    elif algorithm == "mlp-attention":
        # Print a summary of the MLP architecture
        print(model.summary())
        print(separator)
        # We need only the binary labels, not the domain name and the malware family
        y_test_temp = y_test.iloc[:, 1]
        score = model.evaluate(X_test, y_test_temp, verbose=1)
        print(score[0])
        print(score[1])
        print(separator)
    else:
        print("Not Valid algorithm provided")


    return None
