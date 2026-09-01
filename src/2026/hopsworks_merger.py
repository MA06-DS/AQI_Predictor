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
# 2. Load New Feature-Engineered Dataset
# ==========================================

df = pd.read_csv(
    "data/2026/training_dataset_features.csv"
)

print("New dataset shape:", df.shape)

# ==========================================
# 3. Convert datetime column
# ==========================================
df["datetime_local"] = pd.to_datetime(
    df["datetime_local"]
)
df["aqi"] = df["aqi"].astype("float64")

# ==========================================
# 4. Get Feature Store
# ==========================================

fs = project.get_feature_store()

# ==========================================
# 5. Get the Existing Feature Group (v2)
# ==========================================
# Same feature group created on first upload -
# use get_feature_group (not get_or_create) since it
# already exists and we don't want to risk redefining
# primary_key/event_time/schema by accident.

feature_group = fs.get_feature_group(
    name="aqi_training_features",
    version=2
)

print("✅ Retrieved feature group v2")
print("Existing schema:", [f.name for f in feature_group.features])

# ==========================================
# 6. Sanity check: schema match
# ==========================================

existing_cols = set(f.name for f in feature_group.features)
new_cols = set(df.columns)

missing_in_new = existing_cols - new_cols
extra_in_new = new_cols - existing_cols

if missing_in_new:
    print("⚠️ Columns in feature group but missing from new data:", missing_in_new)

if extra_in_new:
    print("⚠️ Columns in new data but not in feature group:", extra_in_new)

# ==========================================
# 7. Insert (append/upsert) New Rows
# ==========================================
# insert() upserts on the primary key (datetime_local).
# Rows with a datetime_local that already exists in the
# feature group will be overwritten, not duplicated -
# new/unseen timestamps are appended.

feature_group.insert(
    df,
    wait=True
)

print("✅ New rows appended to aqi_training_features v2")

# ==========================================
# 8. Final Information
# ==========================================

print("\nRows in this batch:", len(df))
print("Number of Features:", len(df.columns))

print("\nFeature Group:")
print(feature_group)