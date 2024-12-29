from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Initialize Flask App
app = Flask(__name__)

# Load the saved model and preprocessing components
xgb_model = joblib.load('xgb_model.pkl')
encoder = joblib.load('encoder.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON
        input_data = request.get_json()
        df = pd.DataFrame(input_data)

        # Preprocess the input data
        features = ['Heart Rate', 'Temp', 'Humidity', 'CO Amount']
        df['Activity'] = encoder.transform(df['Activity'])
        df[features] = scaler.transform(df[features])

        # Make prediction
        prediction = xgb_model.predict(df)
        response = {
            "AnomalyDetected": bool(prediction[0])
        }

        return jsonify(response), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the app (for local testing)
if __name__ == '__main__':
    app.run(debug=True)
