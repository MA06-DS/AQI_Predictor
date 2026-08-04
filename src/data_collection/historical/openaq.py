import requests
import time
from src.data_collection.config import OPENAQ_API_KEY


class OpenAQClient:
    BASE_URL = "https://api.openaq.org/v3"

    def __init__(self):
        self.headers = {
            "X-API-Key": OPENAQ_API_KEY
        }

    def get_locations(self, latitude, longitude, radius=25000):
        url = f"{self.BASE_URL}/locations"

        params = {
            "coordinates": f"{latitude},{longitude}",
            "radius": radius,
            "limit": 20
        }

        response = requests.get(
            url,
            headers=self.headers,
            params=params
        )

        response.raise_for_status()

        return response.json()

    def get_sensors(self, location_id):
        url = f"{self.BASE_URL}/locations/{location_id}/sensors"

        response = requests.get(url,headers=self.headers)

        response.raise_for_status()

        return response.json()
            
    def get_measurements(self, sensor_id, datetime_from, datetime_to):
        """
        Download all measurements for a sensor between two dates.
        Handles API pagination automatically.
        """

        url = f"{self.BASE_URL}/sensors/{sensor_id}/measurements"

        all_results = []
        page = 1

        while True:

            params = {
                "datetime_from": datetime_from,
                "datetime_to": datetime_to,
                "limit": 1000,
                "page": page
            }

            response = requests.get(
                url,
                headers=self.headers,
                params=params
            )

            response.raise_for_status()

            data = response.json()

            results = data.get("results", [])

            if not results:
                break

            all_results.extend(results)

            print(f"Page {page}: {len(results)} records downloaded")

            # Last page reached
            if len(results) < 1000:
                break

            page += 1

            # Avoid hitting API rate limits
            time.sleep(0.5)

        print(f"\nTotal records downloaded: {len(all_results)}")

        return all_results
        