import mlflow
import os
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from PIL import Image

import mlflow.sklearn

# Function to preprocess images
def preprocess_image(image_path, target_size):
    image = Image.open(image_path)
    image = image.resize(target_size)
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Load the model
model_path = os.path.join('model', 'model.pkl')
model = joblib.load(model_path)

# Load your test data
# Assuming you have a list of image paths and corresponding labels
image_paths= os.path.join('data', 'test')
y_test = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]  # Example labels
# Preprocess images
X_test = np.vstack([preprocess_image(img_path, target_size=(224, 224)) for img_path in image_paths])

# Make predictions
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

# Log metrics to MLflow
with mlflow.start_run():
    mlflow.log_param("model_path", model_path)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Optionally, log the model itself
    mlflow.sklearn.log_model(model, "model")

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")