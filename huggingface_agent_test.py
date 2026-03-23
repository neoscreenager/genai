#import os
from huggingface_hub import InferenceClient

HF_TOKEN = '' #remove before committing, add token to environment varialbe and read it from there


from smolagents import LiteLLMModel
'''
model = LiteLLMModel(
    model_id="meta-llama_-_llama-3.2-1b-instruct",  # Or try other Ollama-supported models
    api_base="http://127.0.0.1:1234",  # Default Ollama local server
    #        num_ctx=8192,
    )
'''
#output = model.text_generation("The capital of france is", max_new_tokens=100,)
client = InferenceClient(provider="hf-inference", model= "meta-llama/Llama-3.3-70B-Instruct",
                         token=HF_TOKEN)
output = client.text_generation(
    "The capital of france is",
    max_new_tokens=100,
    )
print(output)


