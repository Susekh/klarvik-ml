from flask import Flask, request, jsonify
import pandas as pd
import joblib
import random

# Initialize Flask App
app = Flask(__name__)

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
        "Humidity": random.uniform(30, 90),
        "HeatIndex": random.uniform(20, 45),
        "ECG": random.uniform(0.5, 1.5),
        "Heart Rate": 0,  # Placeholder, will be calculated
        "CO": random.uniform(0, 50),
        "Temp": random.uniform(35.5, 40.0),
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
        response = {
            "AnomalyDetected": bool(prediction[0])
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
        response = {
            "Humidity": simulated_data["Humidity"],
            "HeatIndex": simulated_data["HeatIndex"],
            "ECG": simulated_data["ECG"],
            "Heart Rate": simulated_data["Heart Rate"],
            "CO": simulated_data["CO"],
            "Temp": simulated_data["Temp"],
            "AnomalyDetected": bool(prediction[0]),
            "message": "Anomaly Detected!" if bool(prediction[0]) else "All Safe."
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
