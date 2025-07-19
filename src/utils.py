import os
import sys

import numpy as np
import pandas as pd
import dill
from sklearn.metrics import r2_score 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score,confusion_matrix, mean_squared_error

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train,X_test,y_test,models, param):
    try:
        report = {}
        
        for model_name, model in models.items():
            try:
                print(f"\nTraining model: {model_name}")
                hyperparams = param.get(model_name, {})

                # Turn OFF parallelism for stability
                gs = GridSearchCV(
                    model,
                    hyperparams,
                    cv=3,
                    scoring='accuracy',
                    n_jobs=1,              #  Single thread to prevent crashes
                    error_score='raise',   #  Raise actual error instead of masking it
                    refit=True
                )
                print(f"{model_name} → X_train type: {type(X_train)}, shape: {X_train.shape}")
                print(f"{model_name} → y_train type: {type(y_train)}, dtype: {y_train.dtype}, unique: {np.unique(y_train)}")
                
                gs.fit(X_train, y_train)
                if not hasattr(gs.best_estimator_, "fit"):
                    raise Exception(f"{model_name} → GridSearchCV returned an invalid estimator.")
                # Important: check if it's fitted
                try:
                    # Will fail if not fitted
                    _ = gs.best_estimator_.predict(X_train[:5])
                    best_model = gs.best_estimator_
                except Exception as pred_err:
                    raise Exception(f"{model_name} → Model not fitted after GridSearchCV. Check parameters. Error: {pred_err}")

                
                # Predict
                y_train_pred = best_model.predict(X_train)
                y_test_pred = best_model.predict(X_test)
                
                # Accuracy
                train_acc = accuracy_score(y_train, y_train_pred)
                test_acc = accuracy_score(y_test, y_test_pred)

                # Precision
                train_precision = precision_score(y_train, y_train_pred, average='weighted', zero_division=0)
                test_precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)

                # Recall
                train_recall = recall_score(y_train, y_train_pred, average='weighted', zero_division=0)
                test_recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)

                # F1 Score
                train_f1 = f1_score(y_train, y_train_pred, average='weighted', zero_division=0)
                test_f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)

                # MSE
                train_mse = mean_squared_error(y_train, y_train_pred)
                test_mse = mean_squared_error(y_test, y_test_pred)

                # Confusion Matrices
                train_cm = confusion_matrix(y_train, y_train_pred)
                test_cm = confusion_matrix(y_test, y_test_pred)

                # Print training metrics
                print(f"\n{model_name} → Training Metrics:")
                print(f"Accuracy: {train_acc:.4f}")
                print(f"Precision: {train_precision:.4f}")
                print(f"Recall: {train_recall:.4f}")
                print(f"F1 Score: {train_f1:.4f}")
                print(f"MSE: {train_mse:.4f}")
                print("Confusion Matrix:\n", train_cm)

                # Print testing metrics
                print(f"\n {model_name} → Testing Metrics:")
                print(f"Accuracy: {test_acc:.4f}")
                print(f"Precision: {test_precision:.4f}")
                print(f"Recall: {test_recall:.4f}")
                print(f"F1 Score: {test_f1:.4f}")
                print(f"MSE: {test_mse:.4f}")
                print("Confusion Matrix:\n", test_cm)

                print(f"{model_name} accuracy: {test_acc:.4f}")
                report[model_name] = test_acc

            except Exception as model_err:
                print(f"Skipping model '{model_name}' due to error:\n{model_err}")
                continue

        return report
    
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)

