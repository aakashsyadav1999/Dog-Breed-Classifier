import boto3
from botocore.exceptions import NoCredentialsError
from botocore.exceptions import ClientError
import os
import sys
from datetime import datetime
from src.logger import logging
from src.exception import CustomException
from src.entiy.config import AWS_MODEL_UPLOAD_CONFIG
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

class AWS_MODEL_UPLOAD:

    def __init__(self,aws_upload_config:AWS_MODEL_UPLOAD_CONFIG):

        try:

            self.s3 = boto3.resource('s3',
                aws_access_key_id=os.getenv('AWS_KEY'),
                aws_secret_access_key=os.getenv('SECERT_KEY')
            )
            self.aws_upload_config = aws_upload_config
        except Exception as e:
            logging.error(f'Error in AWS_MODEL_UPLOAD: {e}')

    # Function to upload the model file to S3
    def upload_model_file(self, bucket_name):
        try:
            logging.info(f'Uploading {self.aws_upload_config.FINAL_FILE_NAME_MODEL} to {bucket_name}')
        
        except NoCredentialsError:
            logging.error('Credentials not available')
        except ClientError as e:
            logging.error(f'ClientError: {e}')
        except Exception as e:
            logging.error(f'Exception: {e}')

        # Define the upload path on local system
        parent_directory = os.getcwd()
        upload_directory = os.path.join(parent_directory, self.aws_upload_config.MODEL_DIR)
        logging.info(f"This is the local upload directory: {upload_directory}")

        # Check if the local directory exists
        try:
            if not os.path.exists(upload_directory):
                try:
                    os.makedirs(upload_directory)
                    logging.info(f'Created local directory: {upload_directory}')
                except Exception as e:
                    logging.error(f'Failed to create local directory: {e}')
                    return False
        except Exception as e:
            logging.error(f'Issue with upload_directory: {e}')
            return False

        # Get current date and time for folder structure
        current_time = datetime.now().strftime("%Y/%m/%d_%H%M%S")
        s3_folder_path = f"models/{current_time}/"  # Example folder structure on S3

        # Combine the S3 folder path and final file name to form the full key
        s3_key = os.path.join(s3_folder_path, self.aws_upload_config.FINAL_FILE_NAME_MODEL).replace("\\", "/")  # Ensuring compatibility with S3 paths

        try:
            # Upload the file to S3 with the new path (using current date-time folder)
            self.s3.Bucket(bucket_name).upload_file(
                os.path.join(upload_directory, self.aws_upload_config.FINAL_FILE_NAME_MODEL),  # Local file path
                s3_key  # S3 key (destination path in S3)
            )

            logging.info(f'{self.aws_upload_config.FINAL_FILE_NAME_MODEL} is uploaded from {upload_directory}')
        except NoCredentialsError:
            logging.error('Credentials not available')
        except ClientError as e:
            logging.error(f'ClientError: {e}')
        except Exception as e:
            logging.error(f'Error in Uploading File: {e}')