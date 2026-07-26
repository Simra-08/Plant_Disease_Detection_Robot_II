import requests

with open("test.jpg", "rb") as f:
    img_data = f.read()

response = requests.post(
    "http://127.0.0.1:5000/predict",
    data=img_data,
    headers={"Content-Type": "image/jpeg"}
)

print(response.json())