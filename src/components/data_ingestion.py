import os
import pandas as pd
import numpy as np
from datetime import datetime
import time
from src.logger import logging
from src.exception import CustomException
import zipfile


class DataIngestion:

    def __init__(self) -> None:
        self.file_path = os.path.join(os.getcwd())
        print(f": {self.file_path}")

    def read_zip_file(self) -> pd.DataFrame:
        """
        Read the zip file and return the data frame
        """
        try:
            logging.info("Reading the zip file")
            # Reading the zip file
            data_dir = os.path.join(self.file_path, 'data')
            print(f"Data directory: {data_dir}")

            logging.info(f"Data directory: {data_dir}")
            # Correct zip file path construction (point directly to the zip file)
            zip_file_path = os.path.join(data_dir, 'data.zip')

            logging.info(f"Zip file path: {zip_file_path}")
            logging.info(f"Checking if the zip file exists: {os.path.exists(zip_file_path)}")
            # Check if the zip file exists
            if not os.path.exists(zip_file_path):
                raise FileNotFoundError(f"Zip file {zip_file_path} does not exist.")

            logging.info("Extracting the zip file")
            # Extract the zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
                extracted_files = zip_ref.namelist()

            logging.info(f"Files extracted from the zip file: {extracted_files}")
            # Log the number of files extracted
            if not extracted_files:
                logging.info("No files extracted from the zip file.")
        except Exception as e:
            logging.error(f"Error while extracting zip file: {e}")

    def read_count_of_images(self) -> int:
        image_count = 0

        # Correctly log the full path to the zip file
        logging.info(f"Counting images in the zip file located at: {os.path.join(self.file_path, 'data', 'data.zip')}")
        try:
            # Correct zip file path construction (point directly to the zip file)
            zip_file_path = os.path.join(self.file_path, 'data', 'data.zip')

            # Print and log the actual zip file path
            print(f"Zip file path: {zip_file_path}")

            logging.info(f"Zip file path: {zip_file_path}")
            # Check if the zip file exists
            if not os.path.exists(zip_file_path):
                raise FileNotFoundError(f"Zip file {zip_file_path} does not exist.")

            # Open the zip file and count images
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                extracted_files = zip_ref.namelist()
                for file in extracted_files:
                    if file.endswith('.jpg'):
                        image_count += 1
            logging.info(f"Total image count: {image_count}")

            print(f"Total image count: {image_count}")
            return image_count

        except Exception as e:
            logging.error(f"Error while counting images: {e}")
            return 0  # Return 0 in case of error