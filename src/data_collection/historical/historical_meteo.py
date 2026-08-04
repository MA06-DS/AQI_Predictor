import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

from src.data_collection.historical.openmeteo import OpenMeteoClient


class HistoricalWeatherDownloader:

    def __init__(self):
        self.client = OpenMeteoClient()

    def download_all(self):

        LAT = 24.8607
        LON = 67.0011

        start = datetime(2019, 5, 23)
        end = datetime(2025, 3, 4)

        save_folder = "data/historical/weather"

        os.makedirs(save_folder, exist_ok=True)

        current = start

        while current < end:

            next_month = current + relativedelta(months=1)

            if next_month > end:
                next_month = end

            print(f"\nDownloading {current.strftime('%Y-%m')}")

            weather = self.client.get_weather(
                latitude=LAT,
                longitude=LON,
                start_date=current.strftime("%Y-%m-%d"),
                end_date=next_month.strftime("%Y-%m-%d")
            )

            hourly = weather["hourly"]

            df = pd.DataFrame({
                "datetime_local": hourly["time"],
                "temperature": hourly["temperature_2m"],
                "humidity": hourly["relative_humidity_2m"],
                "pressure": hourly["surface_pressure"],
                "wind_speed": hourly["wind_speed_10m"],
                "wind_direction": hourly["wind_direction_10m"],
                "precipitation": hourly["precipitation"],
                "cloud_cover": hourly["cloud_cover"]
            })

            filename = os.path.join(
                save_folder,
                f"karachi_weather_{current.strftime('%Y_%m')}.csv"
            )

            df.to_csv(filename, index=False)

            print(f"Saved {len(df)} rows.")

            current = next_month

        print("\nWeather download completed!")