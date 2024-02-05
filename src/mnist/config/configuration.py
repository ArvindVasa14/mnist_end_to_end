from dataclasses import dataclass
from pathlib import Path
import os
from src.mnist.logger import logging

@dataclass
class DataIngestionConfig:
    train_data_path:str= os.path.join('artifacts','train_data.csv')
    test_data_path:str= os.path.join('artifacts','test_data.csv')
