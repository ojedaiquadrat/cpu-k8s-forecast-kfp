import requests

url = "http://localhost:8080/v2/models/test-24-inference/infer"
payload = {
    "inputs": [
        {
            "name": "input-0",
            "shape": [1, 5, 1],
            "datatype": "FP32",
            "data": [[[0.1], [0.2], [0.3], [0.4], [0.5]]]
        }
    ]
}

response = requests.post(url, json=payload)
print("Status Code:", response.status_code)
print("Response:", response.json())
