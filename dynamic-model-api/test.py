# importing the requests library
import requests

# api-endpoint
URL = "http://127.0.0.1:5000/transformertest"

data = {
    "temperature": 0.5,
    "prompt": "Alice was sleepy",
}


print("OH YEAHHAHAH IT WORKEDDDD AaAAAAaaAAA!")


try:
    response = requests.post(URL, json=data)
    # Print the response
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.json()}")
except:
    print("url not found")
