# import streamlit as st
# import tensorflow as tf
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import pickle
# # Import the specific preprocess_input function you used
# from tensorflow.keras.applications.efficientnet import preprocess_input

# # --- 1. Load Your Saved Model and Class Names ---
# MODEL_PATH = 'celestic.keras'
# CLASS_NAMES_PATH = 'class_names.pkl'

# # ** THE FIX IS HERE **
# # Tell Keras what 'preprocess_input' is when it loads the model
# custom_objects = {"preprocess_input": preprocess_input}

# # Load the trained model with the custom_objects dictionary
# model = tf.keras.models.load_model(
#     MODEL_PATH,
#     custom_objects=custom_objects,
#     compile=False
# )

# # Load the class names
# with open(CLASS_NAMES_PATH, 'rb') as f:
#     class_names = pickle.load(f)

# # --- 2. Image Preprocessing Function ---
# # This function prepares the user's uploaded image to be fed into the model.
# def preprocess_image(image, image_size=256):
#     img = image.resize((image_size, image_size))
#     img_array = tf.keras.utils.img_to_array(img)
#     img_padded = tf.image.resize_with_pad(
#         img_array, image_size, image_size
#     )
#     img_batch = np.expand_dims(img_padded, axis=0)
#     return img_batch


# # --- 3. Streamlit Web App Interface ---
# st.set_page_config(page_title="Constellation Detector", layout="centered")
# st.title("🔭 Constellation Detector")
# st.write("Upload an image of the night sky, and the AI will try to identify the constellation.")

# # File uploader
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Display the uploaded image
#     image = Image.open(uploaded_file)
#     st.image(image, caption='Your Uploaded Image', use_column_width=True)
    
#     # Preprocess the image and make a prediction
#     st.write("")
#     st.write("Classifying...")
    
#     processed_image = preprocess_image(image)
#     prediction = model.predict(processed_image)
#     predicted_class_index = np.argmax(prediction[0])
#     predicted_class_name = class_names[predicted_class_index]
#     confidence = np.max(prediction[0]) * 100
    
#     # Display the result
#     st.success(f"I'm {confidence:.2f}% sure this is **{predicted_class_name}**.")



# -------Code 2----------

# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import pickle
# import base64

# # --- 1. LOAD SAVED MODEL AND CLASS NAMES ---
# MODEL_PATH = 'celestic.keras'
# CLASS_NAMES_PATH = 'class_names.pkl'
# BACKGROUND_IMAGE_PATH = 'background.gif'

# @st.cache_resource
# def load_keras_model():
#     custom_objects = {"preprocess_input": tf.keras.applications.efficientnet.preprocess_input}
#     model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
#     return model

# @st.cache_data
# def load_class_names():
#     with open(CLASS_NAMES_PATH, 'rb') as f:
#         class_names = pickle.load(f)
#     return class_names

# model = load_keras_model()
# class_names = load_class_names()

# # --- 2. IMAGE PREPROCESSING FUNCTION ---
# def preprocess_image(image_pil, image_size=256):
#     img_array = tf.keras.utils.img_to_array(image_pil)
#     shape = tf.cast(tf.shape(img_array)[:-1], tf.float32)
#     long_dim = max(shape)
#     scale = image_size / long_dim
#     new_shape = tf.cast(shape * scale, tf.int32)
#     img_resized = tf.image.resize(img_array, new_shape)
#     img_padded = tf.image.resize_with_pad(img_resized, image_size, image_size)
#     img_batch = np.expand_dims(img_padded, axis=0)
#     return img_batch

# # --- 3. CUSTOM CSS FOR BACKGROUND AND STYLING ---
# @st.cache_data
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_css_background(image_path):
#     bin_str = get_base64_of_bin_file(image_path)
#     page_bg_img = f'''
#     <style>
#     .stApp {{
#         background-image: url("data:image/gif;base64,{bin_str}");
#         background-size: cover;
#     }}
#     [data-testid="stHeader"] {{
#         background-color: rgba(0, 0, 0, 0);
#     }}
#     /* **FIX 3: Change title text color to white** */
#     .main-title h1 {{
#         color: white; 
#     }}
#     </style>
#     '''
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# # --- 4. STREAMLIT APP INTERFACE ---
# st.set_page_config(page_title="CelesticAI", layout="wide")
# set_css_background(BACKGROUND_IMAGE_PATH)

# # --- SIDEBAR SECTION (LEFT PANEL) ---
# st.sidebar.title("CelesticAI Controls")
# st.sidebar.write("Upload an image of the night sky to identify a constellation.")
# uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# # **FIX 1: Display the uploaded image in the sidebar**
# if uploaded_file is not None:
#     st.sidebar.header("Your Uploaded Image")
#     sidebar_image = Image.open(uploaded_file)
#     st.sidebar.image(sidebar_image, use_container_width=True)

# st.sidebar.header("About this Project")
# st.sidebar.info(
#     "This is a deep learning project that uses a pre-trained EfficientNet model to classify images of constellations."
# )


# # --- MAIN PAGE SECTION (RIGHT PANEL) ---
# # Use markdown for the title to apply the CSS class
# st.markdown('<div class="main-title"><h1>🔭 Constellation Detector</h1></div>', unsafe_allow_html=True)
# st.write("---")

# if uploaded_file is None:
#     st.info("Please upload an image using the panel on the left to begin analysis.", icon="🌠")
# else:
#     # **FIX 2: Removed the extra column and container that was causing the blank box**
#     with st.spinner('Analyzing the cosmos...'):
#         # Re-open the file for processing
#         image_for_processing = Image.open(uploaded_file)
#         processed_image = preprocess_image(image_for_processing)
#         prediction = model.predict(processed_image)[0]
#         predicted_class_index = np.argmax(prediction)
#         predicted_class_name = class_names[predicted_class_index]
#         confidence = prediction[predicted_class_index] * 100
    
#     st.header("Prediction Result")
#     st.metric(label="Predicted Constellation", value=predicted_class_name, delta=f"{confidence:.2f}% Confidence")
#     st.progress(int(confidence))
    
#     with st.expander("Show Detailed Probabilities"):
#         for i, class_name in enumerate(class_names):
#             st.write(f"{class_name}: {prediction[i]*100:.2f}%")



# ---------Code3----------
# import streamlit as st
# import tensorflow as tf
# import numpy as np
# from PIL import Image
# import pickle
# import base64
  
# # --- 1. LOAD SAVED MODEL AND CLASS NAMES ---
# MODEL_PATH = 'celestic.keras'
# CLASS_NAMES_PATH = 'class_names.pkl'
# BACKGROUND_IMAGE_PATH = 'background.gif'


# def load_keras_model():
#     custom_objects = {"preprocess_input": tf.keras.applications.efficientnet.preprocess_input}
#     model = tf.keras.models.load_model(MODEL_PATH, custom_objects=custom_objects, compile=False)
#     return model

# @st.cache_data
# def load_class_names():
#     with open(CLASS_NAMES_PATH, 'rb') as f:
#         class_names = pickle.load(f)
#     return class_names

# model = load_keras_model()
# class_names = load_class_names()

# # --- 2. IMAGE PREPROCESSING FUNCTION ---
# def preprocess_image(image_pil, image_size=256):
#     img_array = tf.keras.utils.img_to_array(image_pil)
#     shape = tf.cast(tf.shape(img_array)[:-1], tf.float32)
#     long_dim = max(shape)
#     scale = image_size / long_dim
#     new_shape = tf.cast(shape * scale, tf.int32)
#     img_resized = tf.image.resize(img_array, new_shape)
#     img_padded = tf.image.resize_with_pad(img_resized, image_size, image_size)
#     img_batch = np.expand_dims(img_padded, axis=0)
#     return img_batch

# # --- 3. CUSTOM CSS FOR BACKGROUND AND STYLING ---
# @st.cache_data
# def get_base64_of_bin_file(bin_file):
#     with open(bin_file, 'rb') as f:
#         data = f.read()
#     return base64.b64encode(data).decode()

# def set_css_background(image_path):
#     bin_str = get_base64_of_bin_file(image_path)
#     page_bg_img = f'''
#     <style>
#     .stApp {{
#         background-image: url("data:image/gif;base64,{bin_str}");
#         background-size: cover;
#     }}
#     [data-testid="stHeader"] {{
#         background-color: rgba(0, 0, 0, 0);
#     }}
    
#     /* ** THE NEW, FORCEFUL FIX IS HERE ** */

#     /* Target all common text elements and force their color to white */
#     .main h1, .main h2, .main h3, .main p, .main li, .main label, .main .st-emotion-cache-10trblm {{
#         color: white !important;
#     }}
#     /* Specifically target the metric value and label */
#     [data-testid="stMetricValue"] {{
#         color: white !important;
#     }}
#     [data-testid="stMetricLabel"] p {{
#         color: #d3d3d3 !important; /* Light grey for the small label */
#     }}
#     /* Keep the green 'delta' text for confidence */
#     [data-testid="stMetricDelta"] {{
#         color: rgb(40, 167, 69) !important;
#     }}
#     /* Keep the info box text dark for readability against its light blue background */
#     [data-testid="stInfo"] p {{
#         color: #262730 !important;
#     }}
#     </style>
#     '''
#     st.markdown(page_bg_img, unsafe_allow_html=True)

# # --- 4. STREAMLIT APP INTERFACE ---
# st.set_page_config(page_title="CelesticAI", layout="wide")
# set_css_background(BACKGROUND_IMAGE_PATH)

# # --- SIDEBAR SECTION (LEFT PANEL) ---
# st.sidebar.title("CelesticAI Controls")
# st.sidebar.write("Upload an image of the night sky to identify a constellation.")
# uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     st.sidebar.header("Your Uploaded Image")
#     sidebar_image = Image.open(uploaded_file)
#     st.sidebar.image(sidebar_image, use_container_width=True)

# st.sidebar.header("About this Project")
# st.sidebar.info(
#     "This is a deep learning project that uses a pre-trained EfficientNet model to classify images of constellations."
# )

# # --- MAIN PAGE SECTION (RIGHT PANEL) ---
# st.title("🔭 Constellation Detector")
# st.write("---")

# if uploaded_file is None:
#     st.info("Please upload an image using the panel on the left to begin analysis.", icon="🌠")
# else:
#     with st.spinner('Analyzing the cosmos...'):
#         image_for_processing = Image.open(uploaded_file)
#         processed_image = preprocess_image(image_for_processing)
#         prediction = model.predict(processed_image)[0]
#         predicted_class_index = np.argmax(prediction)
#         predicted_class_name = class_names[predicted_class_index]
#         confidence = prediction[predicted_class_index] * 100
    
#     st.header("Prediction Result")
#     st.metric(label="Predicted Constellation", value=predicted_class_name, delta=f"{confidence:.2f}% Confidence")
#     st.progress(int(confidence))
    
#     with st.expander("Show Detailed Probabilities"):
#         for i, class_name in enumerate(class_names):
#             st.write(f"{class_name}: {prediction[i]*100:.2f}%")



# ------- Code4--using weights--------
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import base64

# --- 1. DEFINE CONSTANTS AND MODEL ARCHITECTURE ---
MODEL_WEIGHTS_PATH = 'celestic_weights.weights.h5'
CLASS_NAMES_PATH = 'class_names.pkl'
BACKGROUND_IMAGE_PATH = 'background.gif'
IMAGE_SIZE = 256
CHANNELS = 3
N_CLASSES = 8 # Make sure this matches the number of constellations you trained on

# This function builds the exact same model structure you used for training
def build_model():
    input_shape = (IMAGE_SIZE, IMAGE_SIZE, CHANNELS)
    
    # --- Base Model (EfficientNetB0) ---
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    # --- Full Model with Preprocessing and Augmentation ---
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Lambda(tf.keras.applications.efficientnet.preprocess_input),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(N_CLASSES, activation='softmax')
    ])
    return model

# --- 2. LOAD WEIGHTS AND CLASS NAMES ---
@st.cache_resource
def load_model_and_weights():
    model = build_model()
    model.load_weights(MODEL_WEIGHTS_PATH)
    return model

@st.cache_data
def load_class_names():
    with open(CLASS_NAMES_PATH, 'rb') as f:
        class_names = pickle.load(f)
    return class_names

model = load_model_and_weights()
class_names = load_class_names()

# --- 3. IMAGE PREPROCESSING FUNCTION ---
def preprocess_image(image_pil, image_size=256):
    img_array = tf.keras.utils.img_to_array(image_pil)
    shape = tf.cast(tf.shape(img_array)[:-1], tf.float32)
    long_dim = max(shape)
    scale = image_size / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img_resized = tf.image.resize(img_array, new_shape)
    img_padded = tf.image.resize_with_pad(img_resized, image_size, image_size)
    img_batch = np.expand_dims(img_padded, axis=0)
    return img_batch

# --- 4. CUSTOM CSS FOR BACKGROUND AND STYLING ---
# (This section remains the same)
@st.cache_data
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_css_background(image_path):
    bin_str = get_base64_of_bin_file(image_path)
    page_bg_img = f'''
    <style>
    .stApp {{
        background-image: url("data:image/gif;base64,{bin_str}");
        background-size: cover;
    }}
    [data-testid="stHeader"] {{
        background-color: rgba(0, 0, 0, 0);
    }}
    .main h1, .main h2, .main h3, .main p, .main li, .main label, .main .st-emotion-cache-10trblm {{
        color: white !important;
    }}
    [data-testid="stMetricValue"] {{
        color: white !important;
    }}
    [data-testid="stMetricLabel"] p {{
        color: #d3d3d3 !important;
    }}
    [data-testid="stMetricDelta"] {{
        color: rgb(40, 167, 69) !important;
    }}
    [data-testid="stInfo"] p {{
        color: #262730 !important;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# --- 5. STREAMLIT APP INTERFACE ---
# (This section remains the same)
st.set_page_config(page_title="CelesticAI", layout="wide")
set_css_background(BACKGROUND_IMAGE_PATH)

# Sidebar
st.sidebar.title("CelesticAI Controls")
st.sidebar.write("Upload an image of the night sky to identify a constellation.")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    st.sidebar.header("Your Uploaded Image")
    sidebar_image = Image.open(uploaded_file)
    st.sidebar.image(sidebar_image, use_container_width=True)
st.sidebar.header("About this Project")
st.sidebar.info("This is a deep learning project...")

# Main Page
st.title("🔭 Constellation Detector")
st.write("---")
if uploaded_file is None:
    st.info("Please upload an image using the panel on the left to begin analysis.", icon="🌠")
else:
    with st.spinner('Analyzing the cosmos...'):
        image_for_processing = Image.open(uploaded_file)
        processed_image = preprocess_image(image_for_processing)
        prediction = model.predict(processed_image)[0]
        predicted_class_index = np.argmax(prediction)
        predicted_class_name = class_names[predicted_class_index]
        confidence = prediction[predicted_class_index] * 100
    
    st.header("Prediction Result")
    st.metric(label="Predicted Constellation", value=predicted_class_name, delta=f"{confidence:.2f}% Confidence")
    st.progress(int(confidence))
    
    with st.expander("Show Detailed Probabilities"):
        for i, class_name in enumerate(class_names):
            st.write(f"{class_name}: {prediction[i]*100:.2f}%")