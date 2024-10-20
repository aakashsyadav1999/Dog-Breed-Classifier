import csv
import sys
import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import zipfile
from zipfile import ZipFile
from glob import glob

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, Dropout, Input
from tensorflow.keras.layers import GlobalAveragePooling2D, BatchNormalization, Activation
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix

from src.logger import logging
from src.exception import CustomException

from src.entiy.config import DATA_TRANSFORMATION, AWS_DOWNLOAD_CRED

from PIL import Image

TF_ENABLE_ONEDNN_OPTS=0

class DataTransformation:


    def __init__(self,data_transformation: DATA_TRANSFORMATION, aws_download_cred: AWS_DOWNLOAD_CRED):

        self.data_transformation = data_transformation
        self.aws_download_cred = aws_download_cred


    # Load data
    def load_data(self):
        """
        Load the data from the given path
        """
        
        file_path = os.path.join(os.getcwd(), self.aws_download_cred.AWS_DOWNLOAD_DATA_DIR, self.aws_download_cred.LABELS_CSV)
        #with open(os.getcwd() + '\\' + self.aws_download_cred.AWS_DOWNLOAD_DATA_DIR + '\\' + self.aws_download_cred.LABELS_CSV, 'r') as file: #for windows
        #with open(os.getcwd() + '//' + self.aws_download_cred.AWS_DOWNLOAD_DATA_DIR + '//' + self.aws_download_cred.LABELS_CSV, 'r') as file:  #for linux
        with open(file_path, 'r') as file:
            logging.info(f"Data loaded successfully")
            data = pd.read_csv(file)
            logging.info(f"Data shape: {data.shape}")
            data['id'] = data["id"].apply(lambda x: x + ".jpg")

        return data
    
    # Load image lables
    def load_images(self):
        """
        Load all .jpg images from the train directory in a more efficient way.
        """
        # Construct the path pattern to match all .jpg images
        train_path = os.path.join(os.getcwd(), self.aws_download_cred.AWS_DOWNLOAD_DATA_DIR, 'train', '*.jpg')
        
        logging.info(f"Loading images from: {train_path}")
        try:
            # Use glob to get a list of all image paths
            image_paths = glob(train_path)

            if not image_paths:
                logging.warning(f"No .jpg images found in: {train_path}")
                return []

            # Use list comprehension to load all images quickly
            images = [Image.open(path).convert('RGB') for path in image_paths]
            logging.info(f"Loaded {len(images)} images successfully.")
            logging.info(f"Length of images: {len(images)}")
            logging.info(f"Images loaded successfully")
            
            return images, train_path
            

        except PermissionError as e:
            logging.error(f"Permission denied: {e.filename}")
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            return []


    # Split data
    def split_data(self, data, RANDOM_STATE=42):
        """
        Generate image data
        """
        logging.info(f"Splitting data into train, test, and validation sets")
        try:
            # Split data
            train_df, test_df = train_test_split(data, 
                                    test_size=0.2, 
                                    random_state=RANDOM_STATE)
            
            train_df, val_df = train_test_split(data, 
                                                test_size=0.2, 
                                                random_state=RANDOM_STATE)
            
            logging.info(f"Data split successfully")
            return train_df, test_df, val_df
            
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            return []

        
    
    # Image generator
    def image_generator(self):
        """
        Generate image data using ImageDataGenerator
        """
        logging.info(f"Generating image data")
        try:
            #Train data generator
            train_datagen = ImageDataGenerator(rescale=1./255)
            #Test data generator
            test_datagen = ImageDataGenerator(rescale=1./255)
            #Validation data generator
            val_datagen = ImageDataGenerator(rescale=1./255)
            logging.info(f"Image data generated successfully")
        
            return train_datagen, test_datagen, val_datagen
        
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            return []
        
    
    # Data generator
    def data_generator(self,train_datagen, test_datagen, val_datagen, train_df, test_df, val_df):

        """
        Generate data using flow_from_dataframe
        """
        logging.info(f"Generating data using flow_from_dataframe")

        try:
            # Path for images
            path_for_image = os.path.join(os.getcwd(), self.aws_download_cred.AWS_DOWNLOAD_DATA_DIR, 'train')

            # Train, test, and validation generators
            train_generator = train_datagen.flow_from_dataframe(train_df, 
                                                                path_for_image, 
                                                                'id', 
                                                                'breed', 
                                                                target_size=self.data_transformation.SIZE, 
                                                                batch_size=self.data_transformation.BATCH_SIZE, 
                                                                class_mode='categorical')
            
            test_generator = test_datagen.flow_from_dataframe(test_df, 
                                                            path_for_image, 
                                                            'id', 
                                                            'breed', 
                                                            target_size=self.data_transformation.SIZE, 
                                                            batch_size=self.data_transformation.BATCH_SIZE, 
                                                            class_mode='categorical')
            
            val_generator = val_datagen.flow_from_dataframe(val_df, 
                                                            path_for_image, 
                                                            'id', 
                                                            'breed', 
                                                            target_size=self.data_transformation.SIZE, 
                                                            batch_size=self.data_transformation.BATCH_SIZE, 
                                                            class_mode='categorical')
            
            logging.info(f"Data generated successfully")
            return train_generator, test_generator, val_generator
        
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            return []
    
    # Base model
    def base_model(self):
        """
        Generate the base model
        """
        # Base model
        try:
            logging.info(f"Generating base model")
            # Input layer
            input_tensor = Input(shape=(self.data_transformation.SIZE[0], self.data_transformation.SIZE[1], 3))
            base_model = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)
            base_model.trainable = False
            logging.info(f"Base model generated successfully")
        
            return input_tensor, base_model
        
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            return []

    # Output layer
    def output_layer(self,data,base_model,input_tensor):
        """
        Generate the output layer
        """
        try:
        
            logging.info(f"Generating output layer")
            # Output layer
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dropout(self.data_transformation.DROPOUT_RATE)(x)
            output = Dense((len(data['breed'].unique())), activation='softmax')(x)

            # Compile model
            model = Model(inputs=input_tensor, outputs=output)
            model.compile(optimizer=Adam(learning_rate=self.data_transformation.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])
            logging.info(f"Output layer generated successfully")
            
            return model
        
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            return []
    
   # Callbacks
    def call_backs(self):
        """
        Generate the callbacks
        """
        try:
            logging.info(f"Generating callbacks")
            
            # Create 'model' directory if it doesn't exist
            model_dir = 'model'
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            
            # Early stopping and checkpoint saving
            early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
            model_checkpoint = ModelCheckpoint(os.path.join(model_dir, 'model.keras'), 
                                               monitor='val_loss', 
                                               save_best_only=True, 
                                               verbose=1)
            logging.info(f"Callbacks generated successfully")
            
            return early_stopping, model_checkpoint
        
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            return []

    def train_model(self, model, train_generator, val_generator, early_stopping, model_checkpoint):
        """
        Train the model
        """
        try:
            logging.info(f"Training model")
            
            # Check if GPU is available and configure memory growth
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                try:
                    tf.config.experimental.set_memory_growth(physical_devices[0], True)
                    print(f"Using GPU: {physical_devices[0]}")
                except RuntimeError as e:
                    print(e)
            else:
                print("No GPU found, using CPU")

            # Train model
            history = model.fit(train_generator,
                                validation_data=val_generator,
                                steps_per_epoch=train_generator.samples // self.data_transformation.BATCH_SIZE,
                                validation_steps=val_generator.samples // self.data_transformation.BATCH_SIZE,
                                epochs=self.data_transformation.EPOCHS,
                                callbacks=[early_stopping, model_checkpoint])
            
            logging.info(f"Model trained successfully")

            # Save the final trained model
            final_model_path = os.path.join('model', 'final_model.keras')
            model.save(final_model_path)
            logging.info(f"Final model saved at {final_model_path}")
        
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            return []
        
    # Model summary
    def model_evaluation(self,model, test_generator):
        """
        Evaluate the model
        """
        try:
            # Evaluate model on test data
            score = model.evaluate(test_generator)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])
            logging.info(f"Model evaluated successfully")
        
        except Exception as e:
            logging.error(f"An unexpected error occurred: {str(e)}")
            return []


    def init_model(self):
        """
        Initialize the model
        """

        # Initialize the model
        data_trasnformation = DataTransformation(aws_download_cred=self.aws_download_cred, data_transformation=self.data_transformation)    

        # Load data
        data = data_trasnformation.load_data()

        # Load image lables
        images, train_path = data_trasnformation.load_images()


        # Split data
        train_df, test_df, val_df = data_trasnformation.split_data(data)

        # Image generator
        train_datagen, test_datagen, val_datagen = data_trasnformation.image_generator()

        # Data generator
        train_generator, test_generator, val_generator = data_trasnformation.data_generator(train_datagen, test_datagen, val_datagen, train_df, test_df, val_df)

        # Base model
        input_tensor, base_model = data_trasnformation.base_model()

        # Output layer
        model = data_trasnformation.output_layer(data,base_model,input_tensor)

        # Callbacks
        early_stopping, model_checkpoint = data_trasnformation.call_backs()

        # Train model
        data_trasnformation.train_model(model,train_generator, val_generator, early_stopping, model_checkpoint)

        # Model summary
        data_trasnformation.model_evaluation(model, test_generator)