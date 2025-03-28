from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load model
model = joblib.load("naive_bayes_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # Serve the frontend

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])[0]
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
