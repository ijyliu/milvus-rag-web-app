
import requests
import json


# Set headers and data for the request, and authentication
headers = {'Content-Type': 'application/json'}
data = {"model": "gemma3:1b", 
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        }


try:
    response = requests.post(url='http://localhost:3000/v1/chat/completions',
                             headers=headers, 
                             data=json.dumps(data), 
                             stream=True)
    response.raise_for_status()  # Raise an exception for bad status codes
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")


print(response.json()['choices'][0]['message'])


