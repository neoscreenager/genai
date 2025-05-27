# Example: reuse your existing OpenAI setup
from openai import OpenAI

# Point to the local server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

completion = client.chat.completions.create(
  model="meta-llama_-_llama-3.2-1b-instruct",
  messages=[
    {"role": "system", "content": "You are an AI Assistant!"},
    {"role": "user", "content": "What is the current temperature in Jaipur,Rajasthan? Today's date is 16th May 2025."}
  ],
  temperature=0.7,
)

print(completion.choices[0].message)
