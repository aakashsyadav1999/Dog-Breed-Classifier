import os
import sys
from src.constants import *
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass


@dataclass
class AWS_DOWNLOAD_CRED:

    def __init__(self):
        
        # Get the parent directory (one level up from the current working directory)
        self.parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        # Define the new directory path outside of the research directory
        self.download_directory = os.path.join(self.parent_directory, AWS_DOWNLOAD_DATA_DIR)
        # AWS Credentials
        self.AWS_KEY = os.getenv('AWS_KEY')
        self.SECRET_KEY = os.getenv('SECRET_KEY')
        self.BUCKET_NAME = BUCKET_NAME

@dataclass
class DATA_TRANSFORMATION : 

    def __init__(self):
        # Size of the image
        self.SIZE = SIZE
        # Batch Size
        self.BATCH_SIZE = BATCH_SIZE
        # Number of classes
        self.NUM_CLASSES = NUM_CLASSES
        # Learning Rate
        self.LEARNING_RATE = LEARNING_RATE
        # Dropout Rate
        self.DROPOUT_RATE = DROPOUT_RATE
        # Number of Epochs
        self.EPOCHS = EPOCHS