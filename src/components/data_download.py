import boto3
import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.entiy.config import AWS_DOWNLOAD_CRED
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class AWS_DOWNLAOD:

    def __init__(self,aws_download_config:AWS_DOWNLOAD_CRED):

        self.s3 = boto3.resource('s3',
            aws_access_key_id=os.getenv('AWS_KEY'),
            aws_secret_access_key=os.getenv('SECERT_KEY')
        )
        self.aws_download_config = aws_download_config

    def download_file(self, bucket_name, file_name, download_path):
        # Define the download directory
        self.aws_download_config.parent_directory
        logging.info(f'Downloading {file_name} from {bucket_name} to {download_path}')

        # Define the download path
        download_directory = self.aws_download_config.download_directory

        if not download_directory:
            mkdir_cmd = f'mkdir -p {download_directory}'
            os.system(mkdir_cmd)
            logging.info(f'Created directory: {download_directory}')
            

        # Check if the directory exists, and if not, create it
        if not os.path.exists(download_directory):
            os.makedirs(download_directory)

        # Download the file from S3 to the new directory
        self.s3.Bucket(bucket_name).download_file(self.aws_download_config.ZIP_FILE_NAME, os.path.join(download_directory, self.aws_download_config.ZIP_FILE_NAME))
        print(f'{file_name} is downloaded to {download_path}')