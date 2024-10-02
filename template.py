import os
import sys
import time
import logging




list_of_files = [

    "src/components/data_ingestion.py",
    "src/components/data_preprocessing.py",
    "src/components/model_trainer.py",
    "src/components/model_evaluation.py",
    "src/constant/__init__.py",
    "src/entiy/__init__.py",
    "src/entiy/config.py",
    "src/pipeline/__init__.py",
    "src/pipeline/training_pipeline.py",
    "src/pipeline/predicting_pipeline.py",
    "src/utils/__init__.py",
    "src/utils/common.py",
    "src/logger/__init__.py",
    "src/exception/__init__.py",
    "src/services/__init__.py",
    "research/reaseach_notebook.ipynb",
    "data/",
    "app.py",
    "main.py",
    "requirements.txt",
    "Dockerfile",
    "docker-compose.yml",
    "setup.py",
]



# Create the directories and files
for filepath in list_of_files:

    # Create the directories and files
    filepath = os.path.join(os.getcwd(), filepath)
    print(filepath)
    filedir, filename = os.path.split(filepath)

    # Create the directories if they do not exist
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    
    # Create the files if they do not exist
    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Created directory: {filedir}")

    # Create the files if they do not exist
    if (not os.path.exists(filepath) or (os.path.exists(filepath) and input(f"{filename} already exists. Do you want to overwrite it? (y/n): ").lower() == "y")):
        with open(filepath, "w") as f:
            f.write("")

        logging.info(f"Created file: {filepath}")

    # Skip creating the file
    else:
        logging.info(f"Skipped creating file: {filepath}")