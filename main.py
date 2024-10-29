# Import the necessary libraries (tested for Python 3.10)
import os.path

from Scripts.Plots.Plotting import *
from Scripts.Preprocessing.Preprocessing import *
from Scripts.evaluate_model import evaluate_model
from Scripts.evaluate_model_on_family import evaluate_model_on_family
from Scripts.train_model import train_model
from Scripts.utils import calculate_and_explain_shap

# True to print debugging outputs, False to silence the program
DEBUG = True
separator = "-------------------------------------------------------------------------"

# Define the number of clusters that will represent the training dataset for SHAP framework (cannot give all training
# samples)
K_MEANS_CLUSTERS = 100

# Define the number of testing samples on which SHAP will derive interpretations
SAMPLES_NUMBER = 300

# Correlation threshold for Pearson correlation. For feature pairs with correlation higher than the threshold,
# one feature is dropped
CORRELATION_THRESHOLD = 0.9

# Families mainly discussed within the paper
paper_families = ["bamital", "conficker", "cryptolocker", "matsnu", "suppobox", "all_DGAs"]
#paper_families = ["matsnu", "suppobox", "all_DGAs"]

# Families considered for SHAP interpretations
families = ["tranco", "bamital", "banjori", "bedep", "chinad", "conficker", "corebot", "cryptolocker", "dnschanger",
            "dyre", "emotet", "gameover", "gozi", "locky", "matsnu", "monerominer", "murofet", "murofetweekly",
            "mydoom", "necurs", "nymaim2", "nymaim", "oderoor", "padcrypt", "pandabanker", "pitou", "proslikefan",
            "pushdo", "pykspa", "qadars", "qakbot", "qsnatch", "ramnit", "ranbyus", "rovnix", "sisron", "sphinx",
            "suppobox", "sutra", "symmi", "tinba", "tinynuke", "torpig", "urlzone", "vidro", "virut", "wd"]

# Dataset to load
base_path = r"C:\Users\nbazo\Desktop\Netmode\fedxai4dga\fedxai4dga"
filename = base_path + r"\Data\Processed\labeled_dataset_features.csv"


if __name__ == "__main__":

    """
    # Load the dataset
    df, features = load_dataset(filename)

    if DEBUG:
        print("Before correlation: The dataframe is:")
        print(df)
        print(separator)
        print("Before correlation: The shape of the dataframe is:")
        print(df.shape)
        print(separator)
        print("Before correlation: The names of the features are:")
        print(features)
        print(separator)

    # Drop features based on Pearson correlation
    to_drop, df, features = drop_features_by_correlation(df)
    print("Dropped features because of correlation: ", str(to_drop))

    if DEBUG:
        print("After correlation: The new dataframe is:")
        print(df)
        print(separator)
        print("After correlation: The shape of the dataframe is:")
        print(df.shape)
        print(separator)
        print("After correlation: The names of the features are:")
        print(features)
        print(separator)

    # Split dataset into training and testing portions
    X_train, y_train, X_test, y_test = split_dataset(df)

    if DEBUG:
        print("Unscaled X_train:")
        print(X_train)
        print(separator)
        print("Size of X_train:")
        print(len(X_train))
        print(separator)
        print("y_train:")
        print(y_train)
        print(separator)
        print("Size of y_train:")
        print(len(y_train))
        print(separator)
        print("Unscaled X_test:")
        print(X_test)
        print(separator)
        print("Size of X_test:")
        print(len(X_test))
        print(separator)
        print("y_test:")
        print(y_test)
        print(separator)
        print("Size of y_test:")
        print(len(y_test))
        print(separator)

    # Scale dataset using min-max scaling
    X_train, X_test = scale_dataset(X_train, X_test)

    if DEBUG:
        print("Scaled X_train:")
        print(X_train)
        print(separator)
        print("Scaled X_test:")
        print(X_test)
        print(separator)

    # Data oversampling to deal with class imbalance
    X_train, y_train = oversample_data(X_train, y_train)

    if DEBUG:
        print("Size of oversampled X_train:")
        print(len(X_train))
        print(separator)
        print("Size of oversampled y_train:")
        print(len(y_train))
        print(separator)
    """

    # ----------------------------------------------------------------------------------load train data
    train_filename = base_path + r'\Data\Processed\train_data.csv'
    train_data, features = load_dataset(train_filename)
    #train_data = train_data.sample(frac=0.2)
    # unsupervised-Anomaly Detection
    # SOS ! to proceed with supervised learning comment out the following line !
    # train_data = train_data[train_data["Label"] == 0]
    # supervised-Classification
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1:]
    print(X_train.shape)
    # ----------------------------------------------------------------------------------load test data
    test_filename = base_path + r'\Data\Processed\test_data.csv'
    test_data, _ = load_dataset(test_filename)
    # test_data = test_data.sample(frac=0.2)
    X_test = test_data.iloc[:, :-3]
    y_test = test_data.iloc[:, -3:]
    print(X_test.shape)

    # ----------------------------------- Split testing dataset into categories based on malware family
    per_category_test = split_testing_dataset_into_categories(X_test, y_test)

    # Keeping the names will prove useful for local explainability (force plots)
    test_sample = {}
    names_sample = {}
    for family in per_category_test.keys():
        if DEBUG:
            print("Processing family: ", family)
        if len(per_category_test[family]) < SAMPLES_NUMBER:
            test_sample[family] = shap.utils.sample(per_category_test[family], len(per_category_test[family]),
                                                    random_state=1452)
        else:
            test_sample[family] = shap.utils.sample(per_category_test[family], SAMPLES_NUMBER, random_state=1452)
        names_sample[family] = test_sample[family].iloc[:, -1]
        test_sample[family] = test_sample[family].iloc[:, 0:-1]

    if DEBUG:
        print(separator)
        print("Test sample dataframe for all:")
        print(test_sample["all"])
        print(test_sample["all"].shape)
        print(separator)
        print("Names sample dataframe for all:")
        print(names_sample["all"])
        print(names_sample["all"].shape)
        print(separator)

    # SHAP will run forever if you give the entire the training dataset.
    # We use k-means to reduce the training dataset into specific centroids
    # background = shap.kmeans(X_train, K_MEANS_CLUSTERS)

    if DEBUG:
        print("Number of k-means clusters:")
        print(K_MEANS_CLUSTERS)
        print(separator)

    # Algorithms to consider for interpretations
    # algorithms = ["xgboost", "mlp","mlp-attention","mlp-attention2","autoencoder"]
    algorithms = ["mlp-attention"]

    # A dictionary to hold trained models
    model_gs = {}
    model_explainer = {}

    result_path = base_path + r"\Results"
    os.makedirs(result_path, exist_ok=True)

    if os.path.exists(result_path):
        print(f"Folder Results/ created successfully!")
    else:
        print(f"Failed to create folder Results/ ")

    for algorithm in algorithms:
        if DEBUG:
            print("Execution for algorithm: \n\n", algorithm)

        os.makedirs(base_path + f"/Results/{algorithm}/", exist_ok=True)
        # Verify the folder creation
        if os.path.exists(base_path + f"/Results/{algorithm}/"):
            print(f"Folder Results/ {algorithm}/ created successfully!")
        else:
            print(f"Failed to create folder Results/ {algorithm}/ ")

        # Train the machine/deep learning model

        model_temp = train_model(X_train, y_train, algorithm, tune=True, save_path=base_path + f"/Results/{algorithm}/")
        model_gs[algorithm] = model_temp

        # Evaluate the machine/deep learning model

        results = evaluate_model(
            model=model_gs[algorithm],
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            algorithm=algorithm,
            metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
            save_path=base_path + f"/Results/{algorithm}/"
        )
        for family in per_category_test.keys():
            # Get accuracy calculations on testing dataset per malware family
            evaluate_model_on_family(model_gs[algorithm], family, test_sample[family], algorithm)

        # We will derive explanations using the Kernel Explainer
        # model_explainer[algorithm] = shap.KernelExplainer(model_gs[algorithm].predict, background)

    """
    # Verify the folder creation

    
    algorithm = "mlp-attention"

    for fam in paper_families:
        try:
            calculate_and_explain_shap(
                family=fam,
                algorithm=algorithm,
                model_gs=model_gs,
                model_explainer=model_explainer,
                test_sample=test_sample,
                names_sample=names_sample,
                base_path=base_path
            )
        except KeyError as e:
            # Handle the KeyError exception
            print(f"KeyError: {e} for family: {fam}")
            # You can log the error or handle it as needed
            # Continue with the next iteration
            continue

        except Exception as e:
            # Optionally handle other exceptions
            print(f"An error occurred: {e} for family: {fam}")
            # Continue with the next iteration
            continue

    for algorithm in algorithms:
        for family in per_category_test.keys():
            try:
                if (algorithm == "xgboost" and (family == "all_DGAs" or family == "tranco")) or (
                        algorithm.startswith("mlp") and family in paper_families):
                    continue

                calculate_and_explain_shap(
                    family=family,
                    algorithm=algorithm,
                    model_gs=model_gs,
                    model_explainer=model_explainer,
                    test_sample=test_sample,
                    names_sample=names_sample,
                    base_path=base_path
                )
            except KeyError as e:
                # Handle the KeyError exception
                print(f"KeyError: {e} for family: {family}")
                # You can log the error or handle it as needed
                # Continue with the next iteration
                continue

            except Exception as e:
                # Optionally handle other exceptions
                print(f"An error occurred: {e} for family: {family}")
                # Continue with the next iteration
                continue
"""