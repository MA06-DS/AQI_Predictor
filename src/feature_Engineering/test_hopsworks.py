import hopsworks
import os

# Replace these with your own values
PROJECT_NAME = "anaskaaqi"
API_KEY = os.getenv("HOPSWORKS_API_KEY")

project = hopsworks.login(
    project=PROJECT_NAME,
    api_key_value=API_KEY
)

print("=" * 50)
print("✅ Connected to Hopsworks successfully!")
print(f"Project Name : {project.name}")
print("=" * 50)