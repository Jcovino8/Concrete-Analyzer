import urllib.request
import zipfile
import os

# Create Week4 data directory
os.makedirs("data/Week4", exist_ok=True)

# URLs for Week 4 data (tensors)
urls = {
    "Positive_tensors": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/labs/Week4/Positive_tensors.zip",
    "Negative_tensors": "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DL0321EN-SkillsNetwork/labs/Week4/Negative_tensors.zip"
}

for label, url in urls.items():
    zip_path = f"data/Week4/{label}.zip"
    extract_path = "data/Week4"

    print(f"📥 Downloading {label}...")
    urllib.request.urlretrieve(url, zip_path)

    print(f"📂 Extracting {label}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

    os.remove(zip_path)

print("✅ Done. Files are in: data/Week4/Positive_tensors and Negative_tensors")