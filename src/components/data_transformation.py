import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer

from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numeric_features = [
                "Debt ratio %",
                "Total debt/Total net worth",
                "Cash/Total Assets",
                "Net worth/Assets",
                "Operating Profit Rate",
                "Interest Expense Ratio"
                ]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numeric_features)
                ]
            )
            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "Bankrupt?"
            input_train = train_df.drop(columns=[target_column])
            target_train = train_df[target_column]

            input_test = test_df.drop(columns=[target_column])
            target_test = test_df[target_column]

            preprocessor = self.get_data_transformer_object()
            input_train_transformed = preprocessor.fit_transform(input_train)
            input_test_transformed = preprocessor.transform(input_test)

            save_object(self.config.preprocessor_obj_file_path, preprocessor)

            train_arr = np.c_[input_train_transformed, np.array(target_train)]
            test_arr = np.c_[input_test_transformed, np.array(target_test)]

            return train_arr, test_arr, self.config.preprocessor_obj_file_path
        
        except Exception as e:
            raise CustomException(e, sys)
