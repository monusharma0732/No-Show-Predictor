# Patient Appointment No-Show Predictor

Predict whether a patient will attend their medical appointment using machine learning.

## 🚀 Project Overview

This project uses a Random Forest Classifier trained on historical medical appointment data to predict no-shows. The model is integrated into both a Flask API and an interactive Streamlit web app.

## 📊 Dataset

* Source: [Kaggle - Medical Appointment No Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments)
* Features:

  * Demographics (Age, Gender, Neighbourhood)
  * Appointment details (ScheduledDay, AppointmentDay)
  * Medical conditions (Hypertension, Diabetes, etc.)
  * Communication (SMS\_received)
  * Target: No-show (Yes/No)

## 🧠 Machine Learning Pipeline

1. Data Cleaning & Feature Engineering
2. One-hot Encoding for categorical variables
3. Class balancing using SMOTE
4. Model training with Random Forest (F1 Score ≈ 0.84)
5. Model evaluation and persistence with `joblib`

## 🖥️ Streamlit UI

Launch the interactive app:

```bash
streamlit run streamlit_app.py
```

## 🔌 Flask API

Run the API locally:

```bash
python flask_api.py
```

Send POST requests to `http://localhost:5000/predict` with input JSON.

## 🗂️ Project Structure

```
no_show_predictor/
├── requirements.txt         # required modules
├── eda_modeling.ipynb       # ML Model
├── streamlit_app.py         # Streamlit frontend
├── flask_api.py             # Flask backend API
├── appointments.csv         # Raw data file
├── rf_model.pkl             # Trained model
├── preprocessor.pkl         # Saved preprocessing pipeline
└── README.md
```

## 🔧 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

Example `requirements.txt`:

```
pandas
numpy
scikit-learn
imblearn
joblib
flask
streamlit
```

## ✅ Future Improvements

* Add XGBoost or LightGBM comparison
* Visual analytics dashboard
* Dockerize for container deployment

## 🙋‍♂️ Author

**Monu Kumar Sharma**

---

*Built as a transition project into Machine Learning Engineering.*
