import os
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd

from src.data_collection.historical.openaq import OpenAQClient


class HistoricalDownloader:

    def __init__(self):
        self.client = OpenAQClient()

    def download_all(self, sensor_id):

        start = datetime(2019, 5, 23)
        end = datetime(2025, 3, 4)

        save_folder = "data/historical/pollutants"
        os.makedirs(save_folder, exist_ok=True)

        current = start

        while current < end:

            next_month = current + relativedelta(months=1)

            # Don't exceed the final date
            if next_month > end:
                next_month = end

            print(f"\nDownloading {current.strftime('%Y-%m')}")

            data = self.client.get_measurements(
                sensor_id=sensor_id,
                datetime_from=current.strftime("%Y-%m-%dT00:00:00Z"),
                datetime_to=next_month.strftime("%Y-%m-%dT00:00:00Z")
            )

            records = []

            for item in data:

                records.append({
                    "datetime_utc": item["period"]["datetimeFrom"]["utc"],
                    "datetime_local": item["period"]["datetimeFrom"]["local"],
                    "pm25": item["value"]
                })

            df = pd.DataFrame(records)

            filename = os.path.join(
                save_folder,
                f"karachi_pm25_{current.strftime('%Y_%m')}.csv"
            )

            df.to_csv(filename, index=False)

            print(f"Saved {len(df)} rows.")

            current = next_month

        print("\nAll historical data downloaded successfully!")