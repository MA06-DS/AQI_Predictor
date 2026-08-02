import pandas as pd

from src.data_collection.historical.openaq import OpenAQClient

client = OpenAQClient()

SENSOR_ID = 23747

data = client.get_measurements(
    sensor_id=SENSOR_ID,
    datetime_from="2019-05-23T00:00:00Z",
    datetime_to="2019-12-31T23:59:59Z",
)

records = []

for item in data["results"]:
    records.append({
        "datetime_utc": item["period"]["datetimeFrom"]["utc"],
        "datetime_local": item["period"]["datetimeFrom"]["local"],
        "pm25": item["value"]
    })

df = pd.DataFrame(records)

print(df.head())

df.to_csv(
    "src\data_collection\historical\pollutants\karachi_pm25_2019.csv",
    index=False
)

print(f"\nDownloaded {len(df)} measurements.")