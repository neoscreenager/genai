import torch
from transformers import pipeline
import os

os.environ['CURL_CA_BUNDLE'] = r"E:\genai\FG6H0E5819900371.crt"
os.environ['REQUESTS_CA_BUNDLE'] = r"E:\genai\FG6H0E5819900371.crt"
#os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# Load the LLaMA model using HuggingFace Transformers pipeline
model_id = "meta-llama/Llama-3.2-1B-Instruct"
pipe = pipeline(
 "text-generation",
 model=model_id,
 torch_dtype=torch.bfloat16, # Use bfloat16 for efficient computation
 device_map="cpu", # Automatically selects available GPU/CPU
)



def generate_response(system_prompt: str, user_prompt: str) -> str:
    """
    Generate a response from the model based on a system prompt and user 
    prompt.

    Parameters:
    - system_prompt (str): The instruction or persona for the model (e.g., 
    "You are a pirate chatbot").
    - user_prompt (str): The actual user query or message to respond to.

    Returns:
    - str: The generated text response from the model.
    """
    # Construct the input message format for the model
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
 
    # Generate output using the pipeline
    outputs = pipe(messages)
 
    # Extract and return the generated text
    return outputs[0]["generated_text"][-1]['content']



from sentence_transformers import SentenceTransformer
import numpy as np
# Load the model once (outside the function) to avoid reloading on each 
# call
embedding_model = SentenceTransformer("TaylorAI/gte-tiny")

def get_sentence_embedding(sentence: str) -> np.ndarray:
    """
    Generate an embedding vector for a given sentence using a preloaded 
    SentenceTransformer model.
    Parameters:
    - sentence (str): The input sentence to encode.
    Returns:
    - np.ndarray: The sentence embedding as a NumPy array.
    """
    # Encode the sentence into a dense vector using the preloaded model
    embedding = embedding_model.encode(sentence)
    return embedding


# The system prompt sets the behavior or persona of the AI
system_prompt = "You are an AI Chatbot!"
# The user prompt is the actual question or input from the user
user_prompt = "Who discovered Penicillin in 1928?"
# Generate a response from the AI using the system and user prompts
response = generate_response(system_prompt, user_prompt)
# Print the response returned by the AI
print(response)
