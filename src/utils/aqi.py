import pandas as pd

def pm25_to_aqi(pm25):

    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return round(
                ((i_high - i_low) / (c_high - c_low))
                * (pm25 - c_low)
                + i_low
            )

    return None


# Load dataset
df = pd.read_csv("data/processed/training_dataset_clean.csv")

# Create AQI column
df["aqi"] = df["pm25"].apply(pm25_to_aqi)

# Display first 10 rows
print(df[["pm25", "aqi"]].head(10))

# Check for missing AQI values
print("\nMissing AQI values:", df["aqi"].isnull().sum())

# Save updated dataset
df.to_csv("data/processed/training_dataset_aqi.csv", index=False)

print("\nAQI column added successfully!")
#print where df[aqi] is null
print("\nRows with missing AQI values:")
print(df[df["aqi"].isnull()])