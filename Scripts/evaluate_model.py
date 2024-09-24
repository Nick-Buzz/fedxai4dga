# Import the necessary libraries (tested for Python 3.8)
import json
import logging
import os
from datetime import datetime
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    classification_report, make_scorer
from sklearn.model_selection import cross_val_score

from Models.mlp_model import MLPModel

# True to print debugging outputs, False to silence the program
DEBUG = True
separator = "-------------------------------------------------------------------------"
# Define the number of clusters that will represent the training dataset
# for SHAP framework (cannot give all training samples)
K_MEANS_CLUSTERS = 100
# Define the number of testing samples on which SHAP will derive interpretations
SAMPLES_NUMBER = 300
# Correlation threshold for Pearson correlation. For feature pairs with correlation
# higher than the threshold, one feature is dropped
CORRELATION_THRESHOLD = 0.9


# Families mainly discussed within the paper
paper_families = ["bamital", "conficker", "cryptolocker", "matsnu", "suppobox", "all_DGAs"]

# Families considered for SHAP interpretations
families = ["tranco", "bamital", "banjori", "bedep", "chinad", "conficker", "corebot", "cryptolocker", "dnschanger",
            "dyre", "emotet", "gameover", "gozi", "locky", "matsnu", "monerominer", "murofet", "murofetweekly",
            "mydoom", "necurs", "nymaim2", "nymaim", "oderoor", "padcrypt", "pandabanker", "pitou", "proslikefan",
            "pushdo", "pykspa", "qadars", "qakbot", "qsnatch", "ramnit", "ranbyus", "rovnix", "sisron", "sphinx",
            "suppobox", "sutra", "symmi", "tinba", "tinynuke", "torpig", "urlzone", "vidro", "virut", "wd"]


def evaluate_model(model, X_train, y_train,  X_test, y_test, algorithm, metrics=None, cv=None, save_path=None):
    logger = logging.getLogger(__name__)
    logger.info(f"Evaluating {algorithm} model")
    try:
        if algorithm in ["xgboost", "mlp", "mlp-attention", "mlp-attention2", "autoencoder"]:
            if algorithm == "xgboost":
                predictions = model.predict(X_test)
            elif "autoencoder" in algorithm:
                predictions = model.predict(X_test)
                predictions = tf.cast(abs(model.loss(X_test, predictions)) > 0.002, tf.int32).numpy()
            else:
                predictions = model.predict(X_test).round()

            y_true = y_test["Label"].values if algorithm == "xgboost" else y_test.iloc[:, 1].values

            results = {}
            if metrics is None:
                metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]


            print(f"\nEvaluation results for {algorithm}:")
            for metric in metrics:
                if metric == "accuracy":
                    results["accuracy"] = accuracy_score(y_true, predictions)
                    print(f"Accuracy: {results['accuracy']:.4f}")
                elif metric == "precision":
                    results["precision"] = precision_score(y_true, predictions, average='weighted')
                    print(f"Precision: {results['precision']:.4f}")
                elif metric == "recall":
                    results["recall"] = recall_score(y_true, predictions, average='weighted')
                    print(f"Recall: {results['recall']:.4f}")
                elif metric == "f1":
                    results["f1"] = f1_score(y_true, predictions, average='weighted')
                    print(f"F1 Score: {results['f1']:.4f}")
                elif metric == "roc_auc":
                    results["roc_auc"] = roc_auc_score(y_true, predictions)
                    print(f"ROC AUC: {results['roc_auc']:.4f}")

            results["confusion_matrix"] = confusion_matrix(y_true, predictions)
            print("\nConfusion Matrix:")
            print(results["confusion_matrix"])

            results["classification_report"] = classification_report(y_true, predictions)
            print("\nClassification Report:")
            print(results["classification_report"])

            # Merge X_train and X_test
            X = pd.concat([X_train, X_test], axis=0, ignore_index=True)

            # Merge y_train and y_test[1]
            # Assuming y_test is a DataFrame and we want the second column
            y = pd.concat([y_train, y_test.iloc[:, 1]], axis=0, ignore_index=True)
            y = y.iloc[:, 0]

            if cv:
                if isinstance(model, tf.keras.Model):
                    # If a Keras model was passed, wrap it in MLPModel
                    wrapped_model = MLPModel()
                    wrapped_model.model = model
                    model = wrapped_model

                cv_scores = cross_val_score(model, X, y, cv=cv)

                results["cross_val_scores"] = cv_scores
                results["cross_val_mean"] = np.mean(cv_scores)
                results["cross_val_std"] = np.std(cv_scores)

                print(f"\nCross-validation scores: {cv_scores}")
                print(f"Mean CV score: {results['cross_val_mean']:.4f} (+/- {results['cross_val_std']:.4f})")

            if save_path:
                try:
                    # Create a filename with timestamp
                    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
                    filename = f"{algorithm}_evaluation_{timestamp}.json"
                    full_path = os.path.join(save_path, filename)

                    # Convert numpy arrays to lists for JSON serialization
                    serializable_results = {}
                    for key, value in results.items():
                        if isinstance(value, np.ndarray):
                            serializable_results[key] = value.tolist()
                        elif isinstance(value, np.float64):
                            serializable_results[key] = float(value)
                        else:
                            serializable_results[key] = value

                    # Save the results to a JSON file
                    with open(full_path, 'w') as f:
                        json.dump(serializable_results, f, indent=4)

                    logger.info(f"Evaluation report saved to {full_path}")
                    print(f"\nEvaluation report saved to {full_path}")
                except Exception as e:
                    logger.error(f"Error saving report: {str(e)}")
                    print(f"\nError saving report: {str(e)}")

            return results

        else:
            raise ValueError("Not a valid algorithm provided")

    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}")
        print(f"\nError during evaluation: {str(e)}")
        return None


# # Basic usage
# results = evaluate_model(model, X_test, y_test, algorithm="xgboost")
#
# # With custom metrics
# custom_metrics = ["accuracy", "precision", "recall"]
# results = evaluate_model(model, X_test, y_test, algorithm="mlp", metrics=custom_metrics)
#
# # With cross-validation
# results = evaluate_model(model, X_test, y_test, algorithm="mlp-attention", cv=5)
#
# # Saving the report
# results = evaluate_model(model, X_test, y_test, algorithm="xgboost", save_path="/path/to/save/directory")
#
# # Full example with all options
# results = evaluate_model(
#     model=your_model,
#     X_test=X_test,
#     y_test=y_test,
#     algorithm="xgboost",
#     metrics=["accuracy", "precision", "recall", "f1", "roc_auc"],
#     cv=5,
#     save_path="/path/to/save/directory"
# )

