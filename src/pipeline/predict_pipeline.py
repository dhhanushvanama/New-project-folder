import os
from pyexpat import features
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message="X does not have valid feature names")

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

            print("Loading model and preprocessor", flush=True)
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("Successfully loaded model and preprocessor", flush=True)

            # Add missing columns with default value 0.0
            expected_cols = preprocessor.feature_names_in_
            for col in expected_cols:
                if col not in features.columns:
                    features[col] = 0.0

            # Ensure column order
            features = features[expected_cols]

            print("Transforming input features...", flush=True)
            data_scaled = preprocessor.transform(features)

            print("Final input to model:")
            print(features.head(1).T, flush=True)   

            print("Predicting", flush=True)
            preds = model.predict(data_scaled)
            print(f"Prediction result: {preds}", flush=True)

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_data_frame(self):
        try:
            return pd.DataFrame([self.data])
        except Exception as e:
            raise CustomException(e, sys)