import os
import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_download import AWS_DOWNLAOD
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.gemini import Gemini

from src.entiy.config import AWS_DOWNLOAD_CRED, DATA_TRANSFORMATION, AWS_MODEL_UPLOAD_CONFIG, GEMINI_ENV_MODEL
from src.services.aws_model_upload import AWS_MODEL_UPLOAD



class TrainingPipeline:

    def __init__(self):

        self.data_download = AWS_DOWNLAOD(aws_download_config=AWS_DOWNLOAD_CRED())
        self.download_config = AWS_DOWNLOAD_CRED()
        self.data_ingestion = DataIngestion()
        self.data_transformation = DataTransformation(data_transformation=DATA_TRANSFORMATION(),aws_download_cred=AWS_DOWNLOAD_CRED())
        self.aws_model_upload = AWS_MODEL_UPLOAD(aws_upload_config=AWS_MODEL_UPLOAD_CONFIG())
        self.upload_config = AWS_MODEL_UPLOAD_CONFIG()
        self.gemini_env_model = Gemini()


    def download_data(self):
        try:
            self.data_download.download_file(self.download_config.BUCKET_NAME, self.download_config.ZIP_FILE_NAME, self.download_config.download_directory)
        except Exception as e:
            logging.error(f"Error while downloading the data: {e}")
            raise (f"Error while downloading the data: {e}")
        

    def ingest_data(self):
        try:
            self.data_ingestion.read_zip_file()
            self.data_ingestion.read_count_of_images()
        except Exception as e:
            logging.error(f"Error while ingesting the data: {e}")
            raise (f"Error while ingesting the data: {e}")
        
    def data_transformation_pipeline(self):
        try:
            self.data_transformation.init_model()

        except Exception as e:
            logging.error(f"Error while transforming the data: {e}")
            raise (f"Error while transforming the data: {e}")
        
    def upload_model(self):
        try:
            self.aws_model_upload.upload_model_file(self.upload_config.BUCKET_NAME)
        except Exception as e:
            logging.error(f"Error while uploading the model: {e}")
            raise (f"Error while uploading the model: {e}")

    def gemini_env_model_fun(self):
        try:
            self.gemini_env_model.initiate_api()
        except Exception as e:
            logging.error(f"Error while running the gemini model: {e}")
            raise (f"Error while running the gemini model: {e}")

    def run_pipeline(self):
        try:
            self.download_data()
            self.ingest_data()
            self.data_transformation_pipeline()
            self.upload_model()
            self.gemini_env_model_fun()
        except Exception as e:
            logging.error(f"Error while running the pipeline: {e}")
            raise (f"Error while running the pipeline: {e}")