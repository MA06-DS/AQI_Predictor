import pandas as pd
df = pd.read_csv("data/processed/training_dataset_features.csv")
print(df["datetime_local"].duplicated().sum())