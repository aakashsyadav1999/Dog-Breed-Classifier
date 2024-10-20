import os
import sys
import requests
from src.logger import logging
from dotenv import load_dotenv, find_dotenv
from src.entiy.config import GEMINI_ENV_MODEL
import google.generativeai as genai
# Load environment variables from the .env file 
load_dotenv(find_dotenv())

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


class Gemini:
    
    def __init__(self):
        
        self.google_api = GEMINI_ENV_MODEL()

    def gemini_api(self):
        """
        Apply gemini api and call it later in prompt section with image input for response.
        """
        try:
            
            self.gemini_api_obj = self.google_api.GEMINI_API_KEY  #get api key from google api     
            genai.configure(api_key=self.gemini_api_obj)     #configure the api key
            gemini_model = genai.GenerativeModel(self.google_api.GEMINI_MODEL) #get the model name from google api
            logging.info(f"Gemini model configured: {gemini_model}")
            
            return gemini_model
        
        except Exception as e:
            logging.error(f"Error in gemini api: {str(e)}")
            return f"Error: {str(e)}"


    def send_to_gemini(self,predicted_breed):
        try:
            model = self.gemini_api() #get the model name from gemini api
            # Prepare the request payload
            prompt = f"The predicted dog breed is {predicted_breed}. Can you tell me more about this breed?"
            
            respose = model.generate_content(prompt)
            
            print(respose.text)
            return respose
            
        except Exception as e:
            return f"Error: {str(e)}"
        
        
    def initiate_api(self):
        """
        Initiate the gemini api
        """
        try:
            model = self.gemini_api()
            
            self.send_to_gemini(predicted_breed="Labrador Retriever")
        except Exception as e:
            return f"Error: {str(e)}"