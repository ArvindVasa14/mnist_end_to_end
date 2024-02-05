import tensorflow as tf
from tensorflow import keras
import numpy as np 
import pandas as pd 
from keras.models import Sequential
from keras.layers import Dense, Flatten, BatchNormalization, Dropout
from keras.losses import SparseCategoricalCrossentropy
import matplotlib.pyplot as plt
from mnist.components.data_ingestion import DataIngestion
from mnist.components.data_transformation import DataTransformation
from mnist.config.configuration import ModelTrainerConfig
from mnist.logger import logging
import pickle

class ModelTrainer:
    def __init__(self):
        self.trainer_config= ModelTrainerConfig

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("spliting training training and test input data")
            x_train, y_train, x_test, y_test= [
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            ]

            x_train= x_train.reshape(x_train.shape[0], 28, 28, 1)
            x_test= x_test.reshape(x_test.shape[0], 28, 28, 1)

            model = Sequential([
                Flatten(input_shape=(28, 28, 1)),
                Dense(128, activation='relu'),
                Dense(64, activation='relu'),
                Dense(10, activation='softmax')  # Assuming this is the output layer for a classification task
            ])

            model.summary()

            model.compile(optimizer= 'adam', loss= keras.losses.sparse_categorical_crossentropy, metrics= ['accuracy'])

            history= model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs= 20)

            with open(self.trainer_config.model_trainer_file_path,'wb') as model_file:
                pickle.dump(model, model_file)

            logging.info("Highest accuracy achieved : "+ str(history.history['val_accuracy'][-1]))
            return history.history['val_accuracy'][-1]

        except Exception as e:
            raise e

if __name__=='__main__':
    obj1= DataIngestion()
    train_path, test_path= obj1.initiate_data_ingestion()

    obj2= DataTransformation()
    train_arr, test_arr,preprocessor_path= obj2.initiate_data_transformation(train_path, test_path)

    obj3= ModelTrainer()
    highest_accuracy= obj3.initiate_model_trainer(train_arr, test_arr)

    print(highest_accuracy)