"""LLM interaction utilities for knowledge graph generation."""
import requests
import json
import re
from src.knowledge_graph.config import load_config


def call_llm(model, user_prompt, api_key, system_prompt=None, max_tokens=1000, temperature=0.2, base_url=None) -> str:
    """
    Call the language model API.

    Args:
        model: The model name to use
        user_prompt: The user prompt to send
        api_key: The API key for authentication
        system_prompt: Optional system prompt to set context
        max_tokens: Maximum number of tokens to generate
        temperature: Sampling temperature
        base_url: The base URL for the API endpoint

    Returns:
        The model's response as a string
    """
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_key}"
    }

    messages = []

    if system_prompt:
        messages.append({
            'role': 'system',
            'content': system_prompt
        })

    messages.append({
        'role': 'user',
        'content': [
            {
                'type': 'text',
                'text': user_prompt
            }
        ]
    })

    payload = {
        'model': model,
        'messages': messages,
        'max_tokens': max_tokens,
        'temperature': temperature
    }

    response = requests.post(
        base_url,
        headers=headers,
        json=payload
    )

    if response.status_code == 200:
        return response.json()['choices'][0]['message']['content']
    else:
        raise Exception(f"API request failed: {response.text}")



if __name__ == "__main__":
    configurations = load_config("/Users/rajesh/Desktop/rajesh/Archive/teaching/agentic_ai/BMGR-MAY2025-GIAI-2/KG_with_unstructure_data/config.toml")
    model = configurations.get("llm").get("model")
    user_prompt = "who are you?"
    api_key = configurations.get("llm").get("api_key")
    system_prompt = "You are helpful assistance to answer any question only if you know and make answer very crisp and clear."
    base_url = configurations.get("llm").get("base_url")
    response = call_llm(
        model=model,
        user_prompt=user_prompt,
        api_key=api_key,
        system_prompt=system_prompt,
        base_url=base_url
    )
    print(response)