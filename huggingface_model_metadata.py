import requests
import json
import os

os.environ['CURL_CA_BUNDLE'] = r"E:\genai\FG6H0E5819900371.crt"
os.environ['REQUESTS_CA_BUNDLE'] = r"E:\genai\FG6H0E5819900371.crt"
# Set your Hugging Face API token
HUGGINGFACE_TOKEN = ""
if not HUGGINGFACE_TOKEN:
    raise ValueError("HUGGINGFACE_TOKEN environment variable not set")

# Define the model name and the Hugging Face API endpoint
model_name = "bert-base-uncased"
api_url = f"https://huggingface.co/api/models/{model_name}"

# Set up headers with the authorization token
headers = {"Authorization": f"Bearer {HUGGINGFACE_TOKEN}"}

# Send a GET request to the API endpoint
response = requests.get(api_url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    # Parse the JSON response
    model_info = json.loads(response.text)
    print(json.dumps(model_info, indent=2))
else:
    print(f"Error: {response.status_code} - {response.text}")
