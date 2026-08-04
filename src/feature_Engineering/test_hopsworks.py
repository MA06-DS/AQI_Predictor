import hopsworks

# Replace these with your own values
PROJECT_NAME = "anaskaaqi"
API_KEY = "meoEfzwtQ1qJf3W5.pWfDoVuRvsVFwnq1r4wmbwlblutJceofeF2BTdzKJ03G3iwRnzerow7Vn07tkQgz"

project = hopsworks.login(
    project="anaskaaqi",
    api_key_value="meoEfzwtQ1qJf3W5.pWfDoVuRvsVFwnq1r4wmbwlblutJceofeF2BTdzKJ03G3iwRnzerow7Vn07tkQgz"
)

print("=" * 50)
print("✅ Connected to Hopsworks successfully!")
print(f"Project Name : {project.name}")
print("=" * 50)