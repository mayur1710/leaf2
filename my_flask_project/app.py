from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained model
model = load_model('your_model.h5')  # Adjust the path to your model file

# Define a route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define a route for handling predictions
@app.route('/predict', methods=['POST'])
def predict():
    # Get the image file from the request
    img_file = request.files['image']

    # Preprocess the image
    img = image.load_img(img_file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.

    # Make predictions
    predictions = model.predict(img_array)
    class_names = ['class1', 'class2', 'class3']  # Adjust with your class names

    # Get the predicted class
    predicted_class = class_names[np.argmax(predictions)]

    # Return the predicted class as JSON response
    return jsonify({'class': predicted_class})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
