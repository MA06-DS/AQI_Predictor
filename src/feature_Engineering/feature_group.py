import pandas as pd
import hopsworks
import os

# ==========================================
# 1. Connect to Hopsworks
# ==========================================

project = hopsworks.login(
    project="anaskaaqi",
    api_key_value=os.getenv("HOPSWORKS_API_KEY")
)

print("✅ Connected to Hopsworks")

# ==========================================
# 2. Load Feature-Engineered Dataset
# ==========================================

df = pd.read_csv(
    "data/processed/training_dataset_features_9aug.csv"
)

print("Dataset Shape:", df.shape)

# ==========================================
# 3. Convert datetime column
# ==========================================

df["datetime_local"] = pd.to_datetime(
    df["datetime_local"]
)

# ==========================================
# 4. Get Feature Store
# ==========================================

fs = project.get_feature_store()

# ==========================================
# 5. Create/Get Feature Group Version 2
# ==========================================

feature_group = fs.get_or_create_feature_group(
    name="aqi_training_features",
    version=2,
    description="Feature engineered AQI dataset for 72-hour forecasting",
    primary_key=["datetime_local"],
    event_time="datetime_local",
    online_enabled=False
)

print("✅ Feature Group Version 2 ready")

# ==========================================
# 6. Insert Data
# ==========================================

feature_group.insert(
    df,
    wait=True
)

print("✅ 72-hour AQI dataset uploaded successfully!")

# ==========================================
# 7. Final Information
# ==========================================

print("\nDataset Shape:", df.shape)
print("Number of Features:", len(df.columns))

print("\nFeature Group:")
print(feature_group)