import requests
from src.data_collection.config import API_KEY

lat = 24.8607
lon = 67.0011

url = (
    f"https://api.openweathermap.org/data/2.5/weather?"
    f"lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
)

response = requests.get(url)

print("Status Code:", response.status_code)
print(response.json())