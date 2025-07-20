import os
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

            print("Loading model and preprocessor...")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            print("Successfully loaded model and preprocessor")

            print("Transforming input features...")
            data_scaled = preprocessor.transform(features)

            print("Predicting...")
            preds = model.predict(data_scaled)
            return preds

        except Exception as e:
            raise CustomException(e, sys)


# class CustomData:
#     def __init__(self,
#                  debt_ratio,
#                  total_debt_net_worth,
#                  cash_total_assets,
#                  net_worth_assets,
#                  operating_profit_rate,
#                  interest_expense_ratio):
#         self.debt_ratio = debt_ratio
#         self.total_debt_net_worth = total_debt_net_worth
#         self.cash_total_assets = cash_total_assets
#         self.net_worth_assets = net_worth_assets
#         self.operating_profit_rate = operating_profit_rate
#         self.interest_expense_ratio = interest_expense_ratio

#     def get_data_as_data_frame(self):
#         try:
#             data = {
#                 "Debt ratio %": [self.debt_ratio],
#                 "Total debt/Total net worth": [self.total_debt_net_worth],
#                 "Cash/Total Assets": [self.cash_total_assets],
#                 "Net worth/Assets": [self.net_worth_assets],
#                 "Operating Profit Rate": [self.operating_profit_rate],
#                 "Interest Expense Ratio": [self.interest_expense_ratio],
#             }
#             return pd.DataFrame(data)
#         except Exception as e:
#             raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 debt_ratio,
                 total_debt_net_worth,
                 cash_total_assets,
                 net_worth_assets,
                 operating_profit_rate,
                 interest_expense_ratio):
        self.debt_ratio = debt_ratio
        self.total_debt_net_worth = total_debt_net_worth
        self.cash_total_assets = cash_total_assets
        self.net_worth_assets = net_worth_assets
        self.operating_profit_rate = operating_profit_rate
        self.interest_expense_ratio = interest_expense_ratio

    def get_data_as_data_frame(self):
        try:
            data = {
                "Debt ratio %": [self.debt_ratio],
                "Total debt/Total net worth": [self.total_debt_net_worth],
                "Cash/Total Assets": [self.cash_total_assets],
                "Net worth/Assets": [self.net_worth_assets],
                "Operating Profit Rate": [self.operating_profit_rate],
                "Interest Expense Ratio": [self.interest_expense_ratio],
            }
            return pd.DataFrame(data)
        except Exception as e:
            raise CustomException(e, sys)
