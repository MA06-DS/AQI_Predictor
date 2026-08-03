import pandas as pd

df = pd.read_csv("data/processed/training_dataset.csv")

print("Shape:", df.shape)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())
df = df.dropna(subset=["pm25"])
print(df["pm25"].isnull().sum())
print(df["datetime_utc"].isnull().sum())
df = df.dropna(subset=["pm25"])

print(df.shape)

print(df.isnull().sum())
df.to_csv(
    "data/processed/training_dataset_clean.csv",
    index=False
)