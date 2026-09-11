from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os
import logging
from functools import wraps
from pydantic import BaseModel, ValidationError
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

# ============ SETUP ============
app = Flask(__name__)
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Config
API_KEY = os.environ.get("API_KEY")

# ============ MODELS ============
model = None
scaler = None

def load_models():
    global model, scaler
    if model is None:
        model = joblib.load('models/energy_model_v1.pkl')
        scaler = joblib.load('models/scaler_v1.pkl')
        logger.info("✅ Models loaded successfully")

FEATURES = [
    "wind_power", "solar_proxy", "heating_degree", "cooling_degree",
    "precipitation", "hour", "month", "is_weekend",
    "price_lag_24", "price_lag_168", "gas_price"
]

# ============ VALIDATION ============
class PredictionRequest(BaseModel):
    wind_power: float
    solar_proxy: float
    heating_degree: float
    cooling_degree: float
    precipitation: float
    hour: int
    month: int
    is_weekend: int
    price_lag_24: float
    price_lag_168: float
    gas_price: float

# ============ AUTHENTICATION ============
def require_api_key(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not API_KEY:
            logger.error("❌ API key not configured")
            return jsonify({"error": "API key not configured"}), 500
        
        key = request.headers.get("X-API-Key")
        if not key:
            logger.warning("⚠️ Missing X-API-Key header")
            return jsonify({"error": "Missing X-API-Key header"}), 401
        
        if key != API_KEY:
            logger.warning(f"❌ Invalid API key attempt")
            return jsonify({"error": "Invalid API key"}), 401
        
        return f(*args, **kwargs)
    return decorated

# ============ ENDPOINTS ============

@app.route('/health', methods=['GET'])
def health():
    """Public health check (no auth required)"""
    return jsonify({
        'status': 'healthy',
        'model_version': 'v1',
        'model_type': 'XGBoost (tuned)',
        'features_required': FEATURES
    }), 200

@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key
def predict():
    """Make a prediction (requires API key)"""
    try:
        # Validate input
        req = PredictionRequest(**request.json)
        logger.info("✅ Prediction request received and validated")
        
        # Load models
        load_models()
        
        # Prepare data
        df = pd.DataFrame([{
            "wind_power": req.wind_power,
            "solar_proxy": req.solar_proxy,
            "heating_degree": req.heating_degree,
            "cooling_degree": req.cooling_degree,
            "precipitation": req.precipitation,
            "hour": req.hour,
            "month": req.month,
            "is_weekend": req.is_weekend,
            "price_lag_24": req.price_lag_24,
            "price_lag_168": req.price_lag_168,
            "gas_price": req.gas_price,
        }])
        
        # Scale and predict
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]
        
        logger.info(f"📊 Prediction made: {prediction:.2f} EUR/MWh")
        
        return jsonify({
            'predicted_price_eur_mwh': float(prediction),
            'model_version': 'v1',
            'status': 'success',
            'input_features': request.json
        }), 200

    except ValidationError as e:
        logger.error(f"❌ Validation error: {e}")
        return jsonify({
            'error': 'Invalid request data',
            'details': e.errors(),
            'status': 'failed'
        }), 400
    
    except Exception as e:
        logger.error(f"❌ Prediction failed: {e}")
        return jsonify({
            'error': str(e),
            'status': 'failed'
        }), 500

@app.route('/features', methods=['GET'])
@require_api_key
def features():
    """Get required features (requires API key)"""
    return jsonify({
        'features': FEATURES,
        'count': len(FEATURES),
        'description': 'Features required for prediction'
    }), 200

# ============ MAIN ============

if __name__ == '__main__':
    print("🚀 Starting Energy Price Prediction API")
    print(f"📊 Model: XGBoost (tuned)")
    print(f"🎯 Features required: {len(FEATURES)}")
    print("\n💡 API Endpoints:")
    print("   GET  /health     - Check if API is running (no auth)")
    print("   POST /predict    - Make a prediction (requires API key)")
    print("   GET  /features   - Get required features (requires API key)")
    print("\n🔐 Rate limits:")
    print("   10 requests/minute for /predict")
    print("   50 requests/hour globally")
    print("   200 requests/day globally")

    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)