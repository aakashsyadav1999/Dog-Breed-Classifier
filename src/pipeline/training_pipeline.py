import os
import sys

from src.exception import CustomException
from src.logger import logging

from src.components.data_download import AWS_DOWNLAOD
from src.components.data_ingestion import DataIngestion

from src.entiy.config import AWS_DOWNLOAD_CRED



class TrainingPipeline:

    def __init__(self):

        self.aws_download_config = AWS_DOWNLAOD(aws_download_config=AWS_DOWNLOAD_CRED())
        self.data_ingestion = DataIngestion()


    def download_data(self):
        try:
            self.aws_download_config.download_file(self.aws_download_config.BUCKET_NAME, 'dogImages.zip', self.aws_download_config.download_directory)
        except Exception as e:
            logging.error(f"Error while downloading the data: {e}")
            raise (f"Error while downloading the data: {e}")
        

    def ingest_data(self):
        try:
            #self.data_ingestion.read_zip_file()
            self.data_ingestion.read_count_of_images()
        except Exception as e:
            logging.error(f"Error while ingesting the data: {e}")
            raise (f"Error while ingesting the data: {e}")
        



    def run_pipeline(self):
        try:
            #self.download_data()
            self.ingest_data()
        except Exception as e:
            logging.error(f"Error while running the pipeline: {e}")
            raise (f"Error while running the pipeline: {e}")