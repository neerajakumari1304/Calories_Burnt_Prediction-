from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import pickle

from balance_data import BALANCING_DATA

app = Flask(__name__)

model = pickle.load(open("calories.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))
feature_names = pickle.load(open("features.pkl", "rb"))
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    age = float(request.form['age'])
    gender = request.form['gender']
    height = float(request.form['height'])
    weight = float(request.form['weight'])
    duration = float(request.form['duration'])
    heart_rate = float(request.form['heart_rate'])
    body_temp = float(request.form['body_temp'])

    gender_male = 1 if gender == "Male" else 0
    input_data = pd.DataFrame([{
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_Male": gender_male
    }])
    input_data = input_data[feature_names]

    # Scale
    input_scaled = scaler.transform(input_data)

    # Predict
    prediction = model.predict(input_scaled)[0]
    return jsonify({"prediction": round(prediction, 2)})

if __name__ == "__main__":
    app.run(debug=True)