# app.py
import numpy as np
import joblib
import base64
import io
import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
from model import GaussianNB  # Import from model.py instead of defining here

app = Flask(__name__)

# Load the trained model
model = joblib.load("gaussian_nb_model.pkl")

def preprocess_image(image):
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image).reshape(1, -1) / 255.0  # Normalize
    return image

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    image_data = base64.b64decode(data["image"])
    image = Image.open(io.BytesIO(image_data))
    
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    
    return jsonify({"prediction": int(prediction[0])})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
