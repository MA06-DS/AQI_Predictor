import pandas as pd

df=pd.read_csv("data/processed/training_dataset_features_9aug.csv")


print(df.shape)
print([c for c in df.columns if c.startswith("target_aqi_")])
print(df[[c for c in df.columns if c.startswith("target_aqi_")]].head())