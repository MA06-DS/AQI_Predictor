import hopsworks
import os

project = hopsworks.login(
    project="anaskaaqi",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)
fs = project.get_feature_store()

feature_view = fs.get_feature_view(
    name="aqi_72_hour_forecast",
    version=2
)

print("✅ Feature View loaded")

X, y = feature_view.training_data(
    description="72-hour AQI forecasting training dataset"
)

print("X shape:", X.shape)
print("y shape:", y.shape)

print("\nInput Features:")
print(X.columns.tolist())

print("\nTarget Columns:")
print(y.columns.tolist())

print("\nFirst 5 X rows:")
print(X.head())

print("\nFirst 5 y rows:")
print(y.head())