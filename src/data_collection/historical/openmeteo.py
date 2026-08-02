import requests


class OpenMeteoClient:

    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

    def get_weather(
        self,
        latitude,
        longitude,
        start_date,
        end_date
    ):

        params = {
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,

            "hourly": ",".join([
                "temperature_2m",
                "relative_humidity_2m",
                "surface_pressure",
                "wind_speed_10m",
                "wind_direction_10m",
                "precipitation",
                "cloud_cover"
            ]),

            "timezone": "Asia/Karachi"
        }

        response = requests.get(
            self.BASE_URL,
            params=params
        )

        response.raise_for_status()

        return response.json()