# main.py
import io
import base64
import pickle
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from PIL import Image

# Import the specific preprocess_input function for EfficientNet
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. Initialize FastAPI App ---
app = FastAPI()

# Mount the 'static' directory to serve files like the background GIF
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up the template directory for the HTML frontend
templates = Jinja2Templates(directory="templates")

# --- 2. Load Model and Class Names ---
MODEL_PATH = 'celestic.keras'
CLASS_NAMES_PATH = 'class_names.pkl'
model = None
class_names = []

try:
    # When loading a Keras model with custom layers (like a Lambda layer),
    # you must provide a dictionary of these custom objects.
    custom_objects = {"preprocess_input": preprocess_input}
    model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects)
    
    # Load the class names from the pickle file
    with open(CLASS_NAMES_PATH, 'rb') as f:
        class_names = pickle.load(f)
        
except Exception as e:
    print(f"Error loading model or class names: {e}")


# --- 3. Prediction Function ---
# This function processes the uploaded image and returns the prediction.
def predict(image_data: bytes):
    try:
        # Open the image data from the uploaded file
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Resize the image to the required input size for the model (256x256)
        img_resized = image.resize((256, 256))
        
        # Convert the image to a NumPy array
        img_array = tf.keras.utils.img_to_array(img_resized)
        
        # Add a batch dimension to match the model's expected input shape (1, 256, 256, 3)
        img_batch = np.expand_dims(img_array, axis=0)
        
        if model and class_names:
            # Get the raw prediction from the model
            prediction_raw = model.predict(img_batch)[0]
            
            # Find the class with the highest probability
            predicted_class_index = np.argmax(prediction_raw)
            predicted_class_name = class_names[predicted_class_index]
            
            # Get the confidence score as a percentage
            confidence = prediction_raw[predicted_class_index] * 100
            
            return {
                "class_name": predicted_class_name,
                "confidence": f"{confidence:.2f}%"
            }
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        
    return None

# --- 4. FastAPI Endpoints ---
# Root endpoint to serve the main page
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Prediction endpoint to handle image uploads
@app.post("/predict/", response_class=HTMLResponse)
async def predict_image(request: Request, file: UploadFile = File(...)):
    if not file.filename:
        return templates.TemplateResponse("index.html", {"request": request})
    
    # Read the image data from the uploaded file
    image_data = await file.read()
    
    # Get the prediction
    prediction_info = predict(image_data)
    
    # Encode the image data into Base64 to display it on the result page
    image_base64 = base64.b64encode(image_data).decode("utf-8")
    
    # Render the page again with the prediction results and the uploaded image
    return templates.TemplateResponse(
        "index.html", 
        {
            "request": request, 
            "prediction": prediction_info, 
            "image_base64": image_base64
        }
    )
