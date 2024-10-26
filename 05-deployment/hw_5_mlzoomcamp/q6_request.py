import requests

url = "http://127.0.0.1:9696/predict"
client = {"job": "management", "duration": 400, "poutcome": "success"}
res = requests.post(url, json=client).json()
print(f"res: {res}")