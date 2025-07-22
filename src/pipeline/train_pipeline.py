import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

from src.exception import CustomException
from src.logger import logging

def main():
    try:
        logging.info("Starting the training pipeline...")
        # Initialize data ingestion
        obj = DataIngestion()
        train_data, test_data = obj.initiate_data_ingestion()

        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initiate_data_transformation(train_data, test_data)

        trainer = ModelTrainer()
        final_accuracy = trainer.initiate_model_trainer(train_arr, test_arr)

        logging.info(f"training completed with final accuracy: {final_accuracy}")

    except Exception as e:
        logging.error(f"Error occurred during training pipeline: {e}")

if __name__ == "__main__":
    main()