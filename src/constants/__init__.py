
import os
from datetime import datetime

TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

ARTIFACTS_DIR = os.path.join("artifacts")
LOGS_DIR = "logs"
LOGS_FILE_NAME = "SIDFC.log"


#AWS Download Data File
AWS_DOWNLOAD_DATA_DIR = "data"
BUCKET_NAME = "dog-breed-dataset"
