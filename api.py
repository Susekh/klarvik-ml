from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
import pandas as pd
import joblib
import random

# Initialize Flask App
app = Flask(__name__)

# Enable CORS for specific domains
CORS(app, origins=["https://klarvik.vercel.app", "http://localhost:5173"])

# Load the saved model and preprocessing components
xgb_model = joblib.load('xgb_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define features
features = ['Humidity', 'HeatIndex', 'ECG', 'Heart Rate', 'CO', 'Temp']

def calculate_heart_rate(ecg_value):
    """Simulate heart rate extraction from ECG."""
    return max(60, min(120, int(ecg_value * 100)))

def generate_random_data():
    """Generate random data for simulation.""" 
    simulated_data = {
        "Humidity": random.uniform(30, 70),  # Normal range
        "HeatIndex": random.uniform(25, 40),  # Normal range
        "ECG": random.uniform(0.8, 1.2),  # Normal ECG range
        "Heart Rate": 0,  # Placeholder, will be calculated
        "CO": random.uniform(0, 20),  # Normal CO levels
        "Temp": random.uniform(36.0, 38.0),  # Normal body temperature
    }
    simulated_data["Heart Rate"] = calculate_heart_rate(simulated_data["ECG"])
    return simulated_data

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        input_data = request.get_json()
        df = pd.DataFrame(input_data)

        # Preprocess the input data
        df[features] = scaler.transform(df[features])

        # Make prediction
        prediction = xgb_model.predict(df)

        # Rarity logic: Make anomaly detection rare (e.g., 5% chance if predicted True)
        anomaly_detected = bool(prediction[0])
        if anomaly_detected and random.random() > 0.05:  # 5% chance of keeping it True
            anomaly_detected = False

        response = {
            "AnomalyDetected": anomaly_detected
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/simulate', methods=['GET'])
def simulate():
    """Simulate anomaly detection using generated data."""
    try:
        # Generate random data
        simulated_data = generate_random_data()

        # Convert to DataFrame for preprocessing
        df = pd.DataFrame([simulated_data])

        # Preprocess the input data
        df[features] = scaler.transform(df[features])

        # Make prediction
        prediction = xgb_model.predict(df)

        # Rarity logic: Make anomaly detection rare (e.g., 5% chance if predicted True)
        anomaly_detected = bool(prediction[0])
        if anomaly_detected and random.random() > 0.05:  # 5% chance of keeping it True
            anomaly_detected = False

        response = {
            "humidity": simulated_data["Humidity"],
            "heatIndex": simulated_data["HeatIndex"],
            "ecg": simulated_data["ECG"],
            "heartRate": simulated_data["Heart Rate"],
            "co": simulated_data["CO"],
            "temp": simulated_data["Temp"],
            "anomalyDetected": anomaly_detected,
            "message": "Anomaly Detected!" if anomaly_detected else "All Safe."
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"message": "App is working fine!"}), 200

# Run the app (for local testing)
if __name__ == '__main__':
    app.run(debug=True)
