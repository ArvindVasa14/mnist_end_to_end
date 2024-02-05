from mnist.logger import logging
import os
import numpy as np
import pandas as pd
import tensorflow as tf 
from tensorflow import keras
from mnist.config.configuration import DataIngestionConfig

class DataIngestion:
    def __init__(self):
        self.ingestion_config= DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("Data Ingestion Started")
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            x_train_flat= x_train.reshape(x_train.shape[0], -1)
            x_test_flat= x_test.reshape(x_test.shape[0], -1)
            x_train_flat.shape

            train_data = pd.DataFrame(data=x_train_flat, columns=[f'pixel_{i}' for i in range(x_train_flat.shape[1])])
            train_data['label'] = y_train

            test_data = pd.DataFrame(data=x_test_flat, columns=[f'pixel_{i}' for i in range(x_test_flat.shape[1])])
            test_data['label'] = y_test
        
            os.makedirs('artifacts', exist_ok=True)
            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            test_data.to_csv(self.ingestion_config.test_data_path, index=False)
            logging.info("Data Ingestion Completed")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        except Exception as e:
            raise e


if __name__=='__main__':
    obj= DataIngestion()
    obj.initiate_data_ingestion()