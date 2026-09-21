import requests
import os
import time

API_URL = "http://localhost:8080/predict"
API_KEY = os.environ.get("API_KEY")

if not API_KEY:
    raise SystemExit("Set API_KEY in your environment first: export API_KEY=your-real-key")

HEADERS = {
    "X-API-Key": API_KEY,
    "Content-Type": "application/json",
}

# First 10: realistic variation within normal ranges (similar spread to training data)
normal_requests = [
    {"wind_power": 4800, "solar_proxy": 750, "heating_degree": 4, "cooling_degree": 0, "precipitation": 0.2, "hour": 9,  "month": 3,  "is_weekend": 0, "price_lag_24": 142, "price_lag_168": 138, "gas_price": 42},
    {"wind_power": 5200, "solar_proxy": 900, "heating_degree": 2, "cooling_degree": 1, "precipitation": 0.0, "hour": 13, "month": 6,  "is_weekend": 0, "price_lag_24": 155, "price_lag_168": 150, "gas_price": 44},
    {"wind_power": 3900, "solar_proxy": 600, "heating_degree": 6, "cooling_degree": 0, "precipitation": 1.1, "hour": 18, "month": 11, "is_weekend": 1, "price_lag_24": 160, "price_lag_168": 148, "gas_price": 47},
    {"wind_power": 6100, "solar_proxy": 1000,"heating_degree": 0, "cooling_degree": 3, "precipitation": 0.0, "hour": 15, "month": 7,  "is_weekend": 0, "price_lag_24": 138, "price_lag_168": 142, "gas_price": 41},
    {"wind_power": 4500, "solar_proxy": 700, "heating_degree": 3, "cooling_degree": 0, "precipitation": 0.4, "hour": 8,  "month": 4,  "is_weekend": 1, "price_lag_24": 145, "price_lag_168": 140, "gas_price": 43},
    {"wind_power": 5500, "solar_proxy": 850, "heating_degree": 1, "cooling_degree": 2, "precipitation": 0.0, "hour": 12, "month": 5,  "is_weekend": 0, "price_lag_24": 150, "price_lag_168": 146, "gas_price": 45},
    {"wind_power": 4200, "solar_proxy": 650, "heating_degree": 7, "cooling_degree": 0, "precipitation": 2.0, "hour": 20, "month": 12, "is_weekend": 0, "price_lag_24": 165, "price_lag_168": 155, "gas_price": 49},
    {"wind_power": 5800, "solar_proxy": 950, "heating_degree": 0, "cooling_degree": 4, "precipitation": 0.0, "hour": 16, "month": 8,  "is_weekend": 1, "price_lag_24": 135, "price_lag_168": 139, "gas_price": 40},
    {"wind_power": 4700, "solar_proxy": 780, "heating_degree": 4, "cooling_degree": 0, "precipitation": 0.3, "hour": 10, "month": 3,  "is_weekend": 0, "price_lag_24": 143, "price_lag_168": 141, "gas_price": 42},
    {"wind_power": 5000, "solar_proxy": 800, "heating_degree": 5, "cooling_degree": 0, "precipitation": 0.5, "hour": 14, "month": 6,  "is_weekend": 0, "price_lag_24": 150, "price_lag_168": 145, "gas_price": 45},
]

# Last 5: deliberately drifted — values well outside typical training ranges,
# simulating a real-world scenario (e.g. an unusual weather event, energy market shock)
drifted_requests = [
    {"wind_power": 9500, "solar_proxy": 1400, "heating_degree": 0, "cooling_degree": 9,  "precipitation": 0.0, "hour": 14, "month": 7,  "is_weekend": 0, "price_lag_24": 210, "price_lag_168": 190, "gas_price": 78},
    {"wind_power": 1200, "solar_proxy": 150,  "heating_degree": 14, "cooling_degree": 0,  "precipitation": 5.5, "hour": 3,  "month": 1,  "is_weekend": 1, "price_lag_24": 260, "price_lag_168": 230, "gas_price": 92},
    {"wind_power": 8800, "solar_proxy": 1300, "heating_degree": 0, "cooling_degree": 10, "precipitation": 0.0, "hour": 15, "month": 8,  "is_weekend": 0, "price_lag_24": 195, "price_lag_168": 205, "gas_price": 85},
    {"wind_power": 900,  "solar_proxy": 100,  "heating_degree": 16, "cooling_degree": 0,  "precipitation": 6.0, "hour": 2,  "month": 1,  "is_weekend": 0, "price_lag_24": 280, "price_lag_168": 250, "gas_price": 95},
    {"wind_power": 9000, "solar_proxy": 1350, "heating_degree": 0, "cooling_degree": 8,  "precipitation": 0.0, "hour": 13, "month": 7,  "is_weekend": 1, "price_lag_24": 220, "price_lag_168": 200, "gas_price": 80},
]

all_requests = normal_requests + drifted_requests

for i, payload in enumerate(all_requests, 1):
    response = requests.post(API_URL, json=payload, headers=HEADERS)
    label = "drifted" if i > 10 else "normal"
    print(f"[{i:2}/{len(all_requests)}] ({label}) status={response.status_code}")
    time.sleep(0.3)  # small delay, avoids tripping the 10/min rate limit too fast

print("\n✅ Done. Check logged_requests.csv for the new rows.")
