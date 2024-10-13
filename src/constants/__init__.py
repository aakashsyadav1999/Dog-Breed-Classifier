
import os
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_DIR = os.path.join("artifacts")
LOGS_DIR = "logs"
LOGS_FILE_NAME = "SIDFC.log"


#AWS Download Data File
AWS_DOWNLOAD_DATA_DIR = "data"
BUCKET_NAME = "dog-breed-dataset"
ZIP_FILE_NAME = "data.zip"
LABELS_CSV = 'labels.csv'


#Data Transformation
SIZE = (350,350)
BATCH_SIZE = 32
LEARNING_RATE = 0.001
DROPOUT_RATE = 0.7
EPOCHS = 5


#Model Upload
MODEL_DIR = "model"
FINAL_FILE_NAME_MODEL = "final_model.keras"
