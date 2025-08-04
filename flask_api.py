# flask_api.py
from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and preprocessor
model = joblib.load('rf_model.pkl')
preprocessor = joblib.load('preprocessor.pkl')

@app.route('/')
def home():
    return "Welcome to the Patient No-Show Predictor API"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_df = pd.DataFrame([data])
    input_encoded = preprocessor.transform(input_df)
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]

    return jsonify({
        'prediction': int(prediction),
        'confidence_no_show': round(probability, 2),
        'confidence_show': round(1 - probability, 2)
    })

if __name__ == '__main__':
    app.run(debug=True)
