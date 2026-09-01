import os

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

if not OPENWEATHER_API_KEY:
    raise ValueError("OPENWEATHER_API_KEY not found.")

if not OPENAQ_API_KEY:
    raise ValueError("OPENAQ_API_KEY not found.")