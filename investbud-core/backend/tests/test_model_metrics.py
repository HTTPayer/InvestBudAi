"""
Test the /model/metrics endpoint
"""
import requests
import json

# Test the endpoint
response = requests.get("http://localhost:8000/model/metrics")

if response.status_code == 200:
    data = response.json()
    print("=" * 70)
    print("MODEL METRICS ENDPOINT RESPONSE")
    print("=" * 70)
    print(json.dumps(data, indent=2))
else:
    print(f"Error: {response.status_code}")
    print(response.text)
