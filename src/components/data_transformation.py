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

from src.entiy.config import DATA_TRANSFORMATION 

from PIL import Image



class DataTransformation:


    def __init__(self,data_transformation: DATA_TRANSFORMATION):

        self.data_transformation = data_transformation


    # Load data
    def load_data(self, path):
        """
        Load the data from the given path
        """
        data = pd.read_csv(path)
        return data
    
    # Load image lables
    def load_image_lables(self, path):
        """
        Load the image lables from the given path
        """
        image = Image.open(path)
        return image
    
    # Split data
    def split_data(self, labels_all, RANDOM_STATE=42):
        """
        Generate image data
        """
        train_df, test_df = train_test_split(labels_all, test_size=0.2, random_state=RANDOM_STATE)
        train_df, val_df = train_test_split(labels_all, test_size=0.2, random_state=RANDOM_STATE)

        return train_df, test_df, val_df
    
    # Image generator
    def image_generator(self):
        """
        Generate image data using ImageDataGenerator
        """
        train_datagen = ImageDataGenerator(rescale=1./255)
        test_datagen = ImageDataGenerator(rescale=1./255)
        val_datagen = ImageDataGenerator(rescale=1./255)

        return train_datagen, test_datagen, val_datagen
    
    # Data generator
    def data_generator(self,train_datagen, test_datagen, val_datagen, train_df, test_df, val_df, path_for_image):

        """
        Generate data using flow_from_dataframe
        """

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
        

        return train_generator, test_generator, val_generator
    
    # Base model
    def base_model(self):
        """
        Generate the base model
        """
        # Base model
        input_tensor = Input(shape=(self.data_transformation.SIZE[0], self.data_transformation.SIZE[1], 3))
        base_model = Xception(weights='imagenet', include_top=False, input_tensor=input_tensor)
        base_model.trainable = False

        return input_tensor, base_model

    # Output layer
    def output_layer(self,base_model):
        """
        Generate the output layer
        """
        # Output layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(self.data_transformation.DROPOUT_RATE)(x)
        output = Dense(self.data_transformation.NUM_CLASSES, activation='softmax')(x)

        # Compile model
        model = Model(inputs=self.data_transformation.input_tensor, outputs=output)
        model.compile(optimizer=Adam(learning_rate=self.data_transformation.LEARNING_RATE), loss='categorical_crossentropy', metrics=['accuracy'])

        return model
    
    # Callbacks
    def call_backs(self):
        """
        Generate the callbacks
        """
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=4, verbose=1)
        model_checkpoint = ModelCheckpoint('model.keras', monitor='val_loss', save_best_only=True, verbose=1)

        return early_stopping, model_checkpoint
    
    def train_model(self,model,train_generator, val_generator, early_stopping, model_checkpoint):
        """
        Train the model
        """
        # Train model
        # Ensure TensorFlow uses the GPU
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
                            steps_per_epoch=train_generator.samples//self.data_transformation.BATCH_SIZE,
                            validation_steps=val_generator.samples//self.data_transformation.BATCH_SIZE,
                            epochs=self.data_transformation.EPOCHS,
                            callbacks=[early_stopping, model_checkpoint],
                            )
        
        # Model summary
        def model_evaluation(self,model, test_generator):
            """
            Evaluate the model
            """
            # Evaluate model on test data
            score = model.evaluate(test_generator)
            print("Test loss:", score[0])
            print("Test accuracy:", score[1])


        def init_model(self):
            """
            Initialize the model
            """

            # Initialize the model
            data_trasnformation = DataTransformation()

            # Load data
            data = data_trasnformation.load_data(self.data_transformation.DATA_PATH)