import numpy as np
import joblib
import base64
import io
from flask import Flask, request, jsonify, render_template
from PIL import Image

# Define GaussianNB class
class GaussianNB:
    def __init__(self):
        self.priors = {}
        self.mean = {}
        self.var = {}

    def fit(self, X, y):
        self.classes = np.unique(y)
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = X_c.mean(axis=0)
            self.var[c] = X_c.var(axis=0) + 1e-6
            self.priors[c] = len(X_c) / len(X)

    def gaussian_pdf(self, x, mean, var):
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return (1 / np.sqrt(2 * np.pi * var)) * exponent

    def predict(self, X):
        predictions = []
        for x in X:
            posteriors = {}
            for c in self.classes:
                prior = np.log(self.priors[c])
                likelihood = np.sum(np.log(self.gaussian_pdf(x, self.mean[c], self.var[c])))
                posteriors[c] = prior + likelihood
            predictions.append(max(posteriors, key=posteriors.get))
        return np.array(predictions)

# Initialize Flask app
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
    app.run(debug=True)