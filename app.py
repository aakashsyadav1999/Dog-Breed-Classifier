from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import pandas as pd
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Constants
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg'}

# Create uploads directory if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Load the Keras model and dog breed labels
model_path = os.path.join('model', 'final_model.keras')
labels_path = os.path.join('data', 'labels.csv')

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")

if not os.path.exists(labels_path):
    raise FileNotFoundError(f"Labels file not found at {labels_path}")

model = load_model(model_path)
labels_df = pd.read_csv(labels_path)
dog_breeds = labels_df['breed'].tolist()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(350, 350))  # Adjust target size to 350x350 as per the model's requirement
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image to the range [0, 1]
    return img_array

# Route to serve the HTML prediction page
@app.route('/predict-page', methods=['GET'])
def predict_page():
    return render_template('predict.html')  # This renders the predict.html template

# API route for handling the prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        img_array = prepare_image(file_path)
        predictions = model.predict(img_array)
        
        if predictions.size == 0:
            return jsonify({'error': 'Prediction failed'}), 500
        
        predicted_breed = dog_breeds[np.argmax(predictions)]
        
        return jsonify({'breed': predicted_breed})
    
    return jsonify({'error': 'Invalid file format. Only .jpg and .jpeg files are allowed.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
