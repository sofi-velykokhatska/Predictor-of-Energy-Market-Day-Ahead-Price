from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load models LAZILY (when first needed, not at startup)
model = None
scaler = None

def load_models():
    global model, scaler
    if model is None:
        model = joblib.load('models/energy_model_v1.pkl')
        scaler = joblib.load('models/scaler_v1.pkl')

FEATURES = [
    "wind_power", "solar_proxy", "heating_degree", "cooling_degree",
    "precipitation", "hour", "month", "is_weekend",
    "price_lag_24", "price_lag_168", "gas_price"
]

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_version': 'v1',
        'model_type': 'XGBoost (tuned)',
        'features_required': FEATURES
    }), 200

@app.route('/predict', methods=['POST'])
def predict():
    try:
        load_models()  # Load only when needed
        data = request.json
        
        missing_features = [f for f in FEATURES if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}',
                'required_features': FEATURES,
                'status': 'failed'
            }), 400
        
        df = pd.DataFrame([{feature: data[feature] for feature in FEATURES}])
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]
        
        return jsonify({
            'predicted_price_eur_mwh': float(prediction),
            'model_version': 'v1',
            'status': 'success',
            'input_features': data
        }), 200
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 400

@app.route('/features', methods=['GET'])
def features():
    return jsonify({
        'features': FEATURES,
        'count': len(FEATURES),
        'description': 'Features required for prediction'
    }), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=False)