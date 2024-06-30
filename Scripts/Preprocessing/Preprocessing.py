import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from imblearn.over_sampling import SMOTE

# Families mainly discussed within the paper
paper_families = ["bamital", "conficker", "cryptolocker", "matsnu", "suppobox", "all_DGAs"]

# Families considered for SHAP interpretations
families = ["tranco", "bamital", "banjori", "bedep", "chinad", "conficker", "corebot", "cryptolocker", "dnschanger", "dyre", "emotet", "gameover", "gozi", "locky", "matsnu", "monerominer", "murofet", "murofetweekly", "mydoom", "necurs", "nymaim2", "nymaim", "oderoor", "padcrypt", "pandabanker", "pitou", "proslikefan", "pushdo", "pykspa", "qadars", "qakbot", "qsnatch", "ramnit", "ranbyus", "rovnix", "sisron", "sphinx", "suppobox", "sutra", "symmi", "tinba", "tinynuke", "torpig", "urlzone", "vidro", "virut", "wd"]

# Dataset to load
# filename = "/content/drive/MyDrive/Netmode/fedxai4dga/\
# labeled_dataset_features.csv"

# True to print debugging outputs, False to silence the program
DEBUG = True
separator = "-------------------------------------------------------------------------"

# Define the number of clusters that will represent the training
# dataset for SHAP framework (cannot give all training samples)
K_MEANS_CLUSTERS = 100

# Define the number of testing samples on which SHAP will derive interpretations
SAMPLES_NUMBER = 300

# Correlation threshold for Pearson correlation. For feature pairs with
# correlation higher than the threshold, one feature is dropped
CORRELATION_THRESHOLD = 0.9


def load_dataset(filename):
    # Load the dataset in the form of a csv
    df = pd.read_csv(filename)
    headers = pd.read_csv(filename, index_col=False, nrows=0).columns.tolist()
    features = headers[0:-3]

    # Return a dataframe and the names of the features
    return df, features


def drop_features_by_correlation(df):
    # Calculate correlation coefficients for pairs of features
    df_for_corr = df.drop(labels = ['Name', 'Label', 'Family'], axis = 1)
    correlation_coeffs = df_for_corr.corr()

    # Keep the upper triangular matrix of correlation coefficients
    upper_tri = correlation_coeffs.where(np.triu(np.ones(correlation_coeffs.shape), k = 1).astype(bool))

    # Drop columns with high correlation (one of the features consisting the pair is dropped, the other is kept)
    to_drop = [column for column in upper_tri.columns if any(abs(upper_tri[column]) >= CORRELATION_THRESHOLD)]

    if DEBUG:
        print("Correlation threshold is:")
        print(CORRELATION_THRESHOLD)
        print(separator)

    df = df.drop(columns=to_drop, inplace=False)
    features = df.columns.tolist()[0:-3]

    # Return the names of the dropped features, the new dataframe and the names of the new features within the
    # feature set
    return to_drop, df, features


def split_dataset(df):
    # Split the dataset into training and testing sets
    train_set, test_set = train_test_split(df, test_size=0.2, random_state = 2345, shuffle = True)

    # Split features from labels (the last three columns are domain name, binary label, malware family)
    X_train = train_set.iloc[:, :-3]
    y_train = train_set.iloc[:, -2:-1]
    X_test = test_set.iloc[:, :-3]
    y_test = test_set.iloc[:, -3:]

    return X_train, y_train, X_test, y_test


def scale_dataset(X_train, X_test):
    # Scale the dataset using min-max scaling
    minimum = X_train.min()
    maximum = X_train.max()
    X_train = (X_train - minimum) / (maximum - minimum)
    X_test = (X_test - minimum) / (maximum - minimum)

    # Return the scaled training and testing datasets
    return X_train, X_test


def oversample_data(X_train, y_train):
    # Over sample the data using SMOTE
    sm = SMOTE(random_state = 42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    return X_train, y_train


def split_testing_dataset_into_categories(X_test, y_test):
    # For SHAP purposes, to derive interpretations per malware family,
    # we will split the testing dataset per malware family
    test_merged = pd.merge(left=X_test, left_index=True, right=y_test, right_index=True, how='inner')

    per_category_test = {}

    # For all DGA's regardless of the malware family
    all_test = test_merged.iloc[:, :-2]
    per_category_test["all"] = all_test

    per_category_test["all_DGAs"] = test_merged[test_merged.iloc[:, -1] != "tranco"].iloc[:, :-2]

    for family in families:
        test = test_merged[test_merged.iloc[:, -1] == str(family)]
        test = test.iloc[:, :-2]
        if len(test) > 0:
            per_category_test[str(family)] = test

    # This will hold testing samples per malware family,
    # e.g. per_category_test["bamital"] holds testing samples for bamital
    return per_category_test

