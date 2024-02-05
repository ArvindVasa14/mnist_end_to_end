from sklearn.compose import ColumnTransformer
from mnist.components.data_ingestion import DataIngestion
from mnist.config.configuration import DataTransformationConfig
from sklearn.preprocessing import StandardScaler
import numpy as np 
import pandas as pd
import pickle
import os
from mnist.logger import logging

class DataTransformation:
    def __init__(self):
        self.transformation_config= DataTransformationConfig

    def get_data_transformer(self):
        try:
            preprocessor= ColumnTransformer(
            transformers=[
                ("Normalize", StandardScaler(), [f'pixel_{i}' for i in range(784)])
                ]
            )
            preprocessor_path= os.path.join('artifacts','preprocessor.pkl')
            with open(preprocessor_path, 'wb') as preprocess_file:
                pickle.dump(preprocessor ,preprocess_file)

            return preprocessor
        
        except Exception as e:
            raise e
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("calling for preprocessor object")
            preprocessor= self.get_data_transformer()
            logging.info("got Preprocess objected")

            logging.info("reading train and test data")
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)

            logging.info("transforming data")

            train_input_features= train_df.drop('label', axis=1)
            test_input_features= test_df.drop('label', axis= 1)

            train_arr= np.c_[preprocessor.fit_transform(train_input_features), np.array(train_df['label'])]
            test_arr= np.c_[preprocessor.transform(test_input_features), np.array(test_df['label'])]

            preprocessor_path= os.path.join('artifacts','preprocessor.pkl')
            with open(preprocessor_path, 'wb') as preprocess_file:
                pickle.dump(preprocessor ,preprocess_file)

            print(train_arr.shape, test_arr.shape)

            return (train_arr, test_arr,preprocessor_path)


        except Exception as e:
            raise e




    
