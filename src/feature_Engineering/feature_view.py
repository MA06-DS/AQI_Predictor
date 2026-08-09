import hopsworks
import os

project = hopsworks.login(
    project="anaskaaqi",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

print("✅ Connected to Hopsworks")


fs = project.get_feature_store()

fg = fs.get_feature_group(
    name="aqi_training_features",
    version=2
)

print("✅ Feature Group loaded")

feature_names = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "cloud_cover",
    "pm25",
    "aqi",
    "hour",
    "day",
    "month",
    "year",
    "day_of_week",
    "is_weekend",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_mean_6",
    "aqi_std_6",
    "aqi_mean_12",
    "aqi_std_12",
    "aqi_mean_24",
    "aqi_std_24"
]

label_names = [
    f"target_aqi_{i}"
    for i in range(1, 73)
]

print("Input features:", len(feature_names))
print("Target columns:", len(label_names))

query = fg.select(
    feature_names + label_names
)

feature_view = fs.get_or_create_feature_view(
    name="aqi_72_hour_forecast",
    version=2,
    description="Feature View for 72-hour AQI forecasting",
    query=query,
    labels=label_names
)

print("\n===================================")
print("✅ FEATURE VIEW CREATED")
print("===================================")
print("Name: aqi_72_hour_forecast")
print("Version: 2")
print("Features:", len(feature_names))
print("Labels:", len(label_names))