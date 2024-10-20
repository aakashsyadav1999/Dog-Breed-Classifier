import os
import sys
from src.constants import *
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


@dataclass
class AWS_DOWNLOAD_CRED:

    def __init__(self):
        
        # Get the parent directory (one level up from the current working directory)
        self.parent_directory = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        # Define the new directory path outside of the research directory
        self.download_directory = os.path.join(self.parent_directory, os.getcwd(), 'data')
        # AWS Credentials
        self.AWS_KEY = os.getenv('AWS_KEY')
        self.SECRET_KEY = os.getenv('SECRET_KEY')
        self.BUCKET_NAME = BUCKET_NAME
        self.ZIP_FILE_NAME = ZIP_FILE_NAME
        self.AWS_DOWNLOAD_DATA_DIR = AWS_DOWNLOAD_DATA_DIR
        self.LABELS_CSV = LABELS_CSV

@dataclass
class DATA_TRANSFORMATION : 

    def __init__(self):
        # Size of the image
        self.SIZE = SIZE
        # Batch Size
        self.BATCH_SIZE = BATCH_SIZE
        # Number of classes
        #self.NUM_CLASSES = NUM_CLASSES
        # Learning Rate
        self.LEARNING_RATE = LEARNING_RATE
        # Dropout Rate
        self.DROPOUT_RATE = DROPOUT_RATE
        # Number of Epochs
        self.EPOCHS = EPOCHS

@dataclass
class AWS_MODEL_UPLOAD_CONFIG:

    def __init__(self):
        # AWS Credentials
        self.AWS_KEY = os.getenv('AWS_KEY')
        # AWS Secret Key
        self.SECRET_KEY = os.getenv('SECRET_KEY')
        #Directory Name for the model
        self.MODEL_DIR = MODEL_DIR
        # Final Model File Name
        self.FINAL_FILE_NAME_MODEL = FINAL_FILE_NAME_MODEL
        # Bucket Name
        self.BUCKET_NAME = BUCKET_NAME
        
        
@dataclass
class GEMINI_ENV_MODEL:

    def __init__(self):
        # Gemini Credentials
        self.GEMINI_API_KEY = GEMINI_API_KEY
        # Gemini Model Name
        self.GEMINI_MODEL = GEMINI_MODEL