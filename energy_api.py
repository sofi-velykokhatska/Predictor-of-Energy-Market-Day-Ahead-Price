from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load the trained model and scaler
model_path = 'models/energy_model_v1.pkl'
scaler_path = 'models/scaler_v1.pkl'

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError(f"Model or scaler not found. Make sure {model_path} and {scaler_path} exist.")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# Feature names - must match training data
FEATURES = [
    "wind_power", "solar_proxy", "heating_degree", "cooling_degree",
    "precipitation", "hour", "month", "is_weekend",
    "price_lag_24", "price_lag_168", "gas_price"
]

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint
    Returns status of the API and model
    """
    return jsonify({
        'status': 'healthy',
        'model_version': 'v1',
        'model_type': 'XGBoost (tuned)',
        'features_required': FEATURES
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Make a prediction for energy price
    
    Expected JSON input:
    {
        "wind_power": 5000,
        "solar_proxy": 800,
        "heating_degree": 5,
        "cooling_degree": 0,
        "precipitation": 0.5,
        "hour": 14,
        "month": 6,
        "is_weekend": 0,
        "price_lag_24": 150,
        "price_lag_168": 145,
        "gas_price": 45
    }
    
    Returns:
    {
        "predicted_price_eur_mwh": 127.54,
        "model_version": "v1",
        "status": "success"
    }
    """
    try:
        data = request.json
        
        # Validate all features are present
        missing_features = [f for f in FEATURES if f not in data]
        if missing_features:
            return jsonify({
                'error': f'Missing features: {missing_features}',
                'required_features': FEATURES,
                'status': 'failed'
            }), 400
        
        # Create DataFrame with features in correct order
        df = pd.DataFrame([{feature: data[feature] for feature in FEATURES}])
        
        # Scale features using the same scaler as training
        df_scaled = scaler.transform(df)
        
        # Make prediction
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
    """
    Get the list of required features
    """
    return jsonify({
        'features': FEATURES,
        'count': len(FEATURES),
        'description': 'Features required for prediction'
    }), 200


if __name__ == '__main__':
    print("🚀 Starting Energy Price Prediction API")
    print(f"📊 Model: XGBoost (tuned)")
    print(f"📁 Model path: {model_path}")
    print(f"📊 Scaler path: {scaler_path}")
    print(f"🎯 Features required: {len(FEATURES)}")
    print("\n💡 API Endpoints:")
    print("   GET  /health     - Check if API is running")
    print("   POST /predict    - Make a prediction")
    print("   GET  /features   - Get required features")
    print("\n🌐 Starting server at http://0.0.0.0:5000")
    
    app.run(host='0.0.0.0', port=8080, debug=False)
