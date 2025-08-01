import os
import io
import base64
from flask import Flask, request, render_template, send_from_directory
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# --- Configuration ---
app = Flask(__name__)

# --- Model Loading ---
try:
    model = load_model('true_model_version_1_constellation.keras')
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

CLASS_NAMES = ['CanisMajor', 'Cassiopeia', 'Crux', 'Leo', 'Orion', 'Scorpius', 'UrsaMajor', 'UrsaMinor']

# --- Image Preprocessing ---
def preprocess_image(image_bytes):
    """
    Loads an image from bytes, resizes it, and prepares it for the model.
    """
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((256, 256))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Flask Routes ---
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', prediction_data=None)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('index.html', error="Model is not available. Please check server logs.")

    if 'file' not in request.files:
        return render_template('index.html', error="No file part in the request.")
    
    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', error="No image selected for uploading.")

    if file:
        img_bytes = file.read()
        processed_image = preprocess_image(img_bytes)
        predictions = model.predict(processed_image)
        
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = CLASS_NAMES[predicted_class_index]
        confidence = np.max(predictions[0]) * 100

        encoded_image = base64.b64encode(img_bytes).decode('utf-8')
        image_data_url = f"data:image/jpeg;base64,{encoded_image}"

        all_predictions = [
            {"class_name": CLASS_NAMES[i], "confidence": predictions[0][i] * 100}
            for i in range(len(CLASS_NAMES))
        ]
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

        prediction_data = {
            'predicted_class': predicted_class_name,
            'confidence': f"{confidence:.2f}",
            'image_data_url': image_data_url,
            'all_predictions': all_predictions
        }

        return render_template('index.html', prediction_data=prediction_data)

    return render_template('index.html', error="An unexpected error occurred.")

# This route is still needed for the background.gif
@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

# The if __name__ == '__main__' block is removed, as Gunicorn will run the app.
