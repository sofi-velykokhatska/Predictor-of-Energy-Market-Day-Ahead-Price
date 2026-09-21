import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

FEATURES = [
    "wind_power", "solar_proxy", "heating_degree", "cooling_degree",
    "precipitation", "hour", "month", "is_weekend",
    "price_lag_24", "price_lag_168", "gas_price"
]

# Reference dataset: the actual data the model was trained on
reference_data = pd.read_csv("reference_data.csv")

# Current dataset: real incoming requests logged by the live API
current_data = pd.read_csv("logged_requests.csv")

# Keep only the feature columns (drop the timestamp column from logged_requests.csv,
# since it isn't a model feature and shouldn't be compared for drift)
current_data = current_data[FEATURES]
reference_data = reference_data[FEATURES]

print(f"Reference dataset: {len(reference_data)} rows (training data)")
print(f"Current dataset:   {len(current_data)} rows (live logged requests)")

report = Report([DataDriftPreset()])
my_eval = report.run(current_data=current_data, reference_data=reference_data)

my_eval.save_html("drift_report.html")
print("\n✅ Drift report saved to drift_report.html — open it in a browser to view.")
