# 

import requests
import json

API_KEY = "1e566909a522ef4cf8c876efeed0e3ed0be5a3d9788de49b01192959a56f900b"

lat = 24.8607
lon = 67.0011

url = "https://api.openaq.org/v3/locations"

headers = {
    "X-API-Key": API_KEY
}

params = {
    "coordinates": f"{lat},{lon}",
    "radius": 25000,   # 25 km
    "limit": 100
}

response = requests.get(url, headers=headers, params=params)

if response.status_code == 200:
    data = response.json()

    print(json.dumps(data, indent=4))

else:
    print("Error:", response.status_code)
    print(response.text)