"""
LLM interaction for generating responses.
"""

import requests
from config import LLM_API_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_TOP_P, LLM_TOP_K, LLM_MAX_TOKENS


def generate_response(prompt):
    """
    Generate a response from the LLM using the given prompt.
    
    Args:
        prompt (str): The prompt to send to the LLM
        
    Returns:
        str: The generated response
    """
    try:
        payload = {
            "model": LLM_MODEL,
            "prompt": prompt,
            "stream": False,
            "temperature": LLM_TEMPERATURE,
            "top_p": LLM_TOP_P,
            "top_k": LLM_TOP_K,
            "max_tokens": LLM_MAX_TOKENS,
        }
        
        result = requests.post(LLM_API_URL, json=payload, timeout=30)
        
        # Check for HTTP errors
        result.raise_for_status()
        
        # Parse response
        response_json = result.json()
        response_text = response_json.get('response')
        
        if not response_text:
            print("Warning: Empty response received from LLM")
            
        return response_text
    
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error occurred: {e}")
        return f"Error generating response: HTTP error {e.response.status_code}"
    
    except requests.exceptions.ConnectionError:
        print("Error: Cannot connect to LLM API. Is the server running?")
        return "Error: Cannot connect to LLM API. Please check if the server is running."
    
    except requests.exceptions.Timeout:
        print("Error: Request to LLM API timed out")
        return "Error: Request timed out. The LLM is taking too long to respond."
    
    except Exception as e:
        print(f"Unexpected error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"