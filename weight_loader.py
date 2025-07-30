import tensorflow as tf

# Define the path to your existing, fully trained model
MODEL_PATH = r'C:\Users\BIT\OneDrive\Orion world\OrionWorld\MLProjects\Celestic_AI\celestic.keras'

# Define the path for the new weights file
WEIGHTS_SAVE_PATH = 'celestic_weights.weights.h5'

print(f"Loading full model from: {MODEL_PATH}")

# Load the complete model
# We need to provide the custom object for the preprocessing layer
custom_objects = {"preprocess_input": tf.keras.applications.efficientnet.preprocess_input}
model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects=custom_objects
)

print("Model loaded successfully.")

# Save only the weights from the loaded model
model.save_weights(WEIGHTS_SAVE_PATH)

print(f"Successfully saved model weights to: {WEIGHTS_SAVE_PATH}")