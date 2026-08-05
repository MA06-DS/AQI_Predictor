import pandas as pd
import hopsworks
import os

# -----------------------------
# Connect to Hopsworks
# -----------------------------
project = hopsworks.login(
    project="anaskaaqi",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

# -----------------------------
# Read Dataset
# -----------------------------
df = pd.read_csv("data/processed/training_dataset_features.csv")

# Convert datetime column
df["datetime_local"] = pd.to_datetime(df["datetime_local"])

print("Dataset Shape:", df.shape)
print("Duplicate datetime_local:", df["datetime_local"].duplicated().sum())

# -----------------------------
# Feature Store
# -----------------------------
fs = project.get_feature_store()

feature_group = fs.get_or_create_feature_group(
    name="aqi_training_features_v2",
    version=1,
    description="AQI training dataset",
    primary_key=["datetime_local"],
    event_time="datetime_local",
    online_enabled=False
)
feature_group.insert(df.iloc[[0]], wait=True)
print("✅ Feature Group uploaded successfully!")