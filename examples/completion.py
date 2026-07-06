"""Example: send a chat completion request and print the assistant response."""

import json
import requests
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("SURF_API_KEY")
headers = {"X-API-KEY": api_key, "Content-Type": "application/json"}
willma_base_url = os.getenv("SURF_API_BASE")
model = os.getenv("LLM_MODEL")

print(f"----------- Using model: {model} -----------\n")

response = requests.post(
    f"{willma_base_url}/chat/completions",
    data=json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": "Hello! What can you do?"}],
    }),
    headers=headers
).json()

try:
    print(response["choices"][0]["message"]["content"])
except Exception as e:
    print(f"Error: {e}")
    print(f"Original response: {response}")
print("\n----------- Done -----------")
