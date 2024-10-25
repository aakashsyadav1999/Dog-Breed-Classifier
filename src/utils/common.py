import pymssql
import json
import os
import sys
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())
class Mysql:
    def __init__(self):
        self.conn = pymssql.connect(
            server=os.getenv('MSSQL_SERVER'),
            user=os.getenv('MSSQL_USER'),
            password=os.getenv('MSSQL_PASSWORD'),
            database=os.getenv('MSSQL_DATABASE'),
            )
        
        
    def insert(self, data):
        cursor = self.conn.cursor()
        try:
            with open(data[0], 'rb') as file:
                binary_data = file.read()
            cursor.execute('INSERT INTO main_data (dog_image, dog_predicted_name, gemini_response) VALUES (%s, %s, %s)', (binary_data, data[1], data[2]))
            self.conn.commit()
        except Exception as e:
            print(f"Error: {e}")
        finally:
            cursor.close()