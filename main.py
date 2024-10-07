import os
import sys
from src.logger import logging
from src.exception import CustomException
from src.pipeline.training_pipeline import TrainingPipeline


def training():

    try:
        # Initialize the training pipeline
        pipeline = TrainingPipeline()

        # Run the training pipeline
        pipeline.run_pipeline()
    
    except Exception as e:
        logging.error(f"Error during training: {e}")
        raise (f"Error during training: {e}")

if __name__ == '__main__':
    training()