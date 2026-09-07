from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
from functools import wraps

app = Flask(__name__)

API_KEY = os.environ.get("API_KEY")

def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        key = request.headers.get("X-API-Key")
        if not API_KEY or key != API_KEY:
            return jsonify({"error": "Unauthorized"}), 401
        return f(*args, **kwargs)
    return decorated

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
@require_api_key
def predict():
    try:
        load_models()
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
@require_api_key
def features():
    return jsonify({
        'features': FEATURES,
        'count': len(FEATURES),
        'description': 'Features required for prediction'
    }), 200

if __name__ == '__main__':
    print("🚀 Starting Energy Price Prediction API")
    print(f"📊 Model: XGBoost (tuned)")
    print(f"🎯 Features required: {len(FEATURES)}")
    print("\n💡 API Endpoints:")
    print("   GET  /health     - Check if API is running")
    print("   POST /predict    - Make a prediction")
    print("   GET  /features   - Get required features")

    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)