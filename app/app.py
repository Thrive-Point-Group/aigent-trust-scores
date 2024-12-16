from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import numpy as np
from typing import List, Dict
import os
import time
from functools import wraps
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app, resources={
    r"/calculate-trust": {
        "origins": [
            "https://higherrrrrrr.fun",
            "http://localhost:3000",
            "http://localhost:8080"
        ],
        "methods": ["POST"],
        "allow_headers": ["Content-Type", "X-API-Key"]
    }
})

# OpenRouter configuration
openai.api_base = "https://openrouter.ai/api/v1"
DEFAULT_API_KEY = os.getenv('OPENROUTER_API_KEY')

# Rate limiting setup
last_request_time = 0
RATE_LIMIT_SECONDS = 5

def rate_limit_if_default_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        global last_request_time
        
        # Get the API key from the request headers
        api_key = request.headers.get('X-API-Key', DEFAULT_API_KEY)
        
        # Only apply rate limiting if using the default API key
        if api_key == DEFAULT_API_KEY:
            current_time = time.time()
            time_since_last_request = current_time - last_request_time
            
            if time_since_last_request < RATE_LIMIT_SECONDS:
                wait_time = RATE_LIMIT_SECONDS - time_since_last_request
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'wait_seconds': wait_time
                }), 429
            
            last_request_time = current_time
        
        return f(*args, **kwargs)
    return decorated_function

def calculate_perplexity(messages: List[Dict], output: str, api_key: str) -> float:
    """Calculate perplexity for a given set of messages and output using token probabilities."""
    try:
        openai.api_key = api_key
        client = openai.OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )

        # Create full message array including the output
        full_messages = messages + [{"role": "assistant", "content": output}]
        
        response = client.chat.completions.create(
            model="openai/gpt-4-turbo",
            messages=full_messages,
            temperature=0,
            logprobs=True,
            max_tokens=0
        )
        
        # Get logprobs from the response
        token_logprobs = response.choices[0].logprobs.token_logprobs
        # Remove None values that might appear at the start
        token_logprobs = [lp for lp in token_logprobs if lp is not None]
        
        # Calculate perplexity
        avg_log_prob = np.mean(token_logprobs)
        perplexity = np.exp(-avg_log_prob)
        
        return perplexity
    
    except Exception as e:
        print(f"Error calculating perplexity: {e}")
        return float('inf')

def calculate_trust_score(perplexity: float) -> dict:
    """Convert perplexity to a trust score between 0 and 1 with classification."""
    # Lower perplexity means higher trust
    # Using a simple inverse exponential transformation
    numeric_score = np.exp(-perplexity / 100)  # Adjust the scaling factor as needed
    score = min(max(numeric_score, 0), 1)  # Clamp between 0 and 1
    
    # Classify the trust level
    if score >= 0.8:
        classification = "HIGH"
    elif score >= 0.5:
        classification = "MEDIUM"
    else:
        classification = "LOW"
    
    return {
        "score": score,
        "classification": classification,
        "description": get_trust_description(classification)
    }

def get_trust_description(classification: str) -> str:
    """Get a human-readable description of the trust classification."""
    descriptions = {
        "HIGH": "The response appears highly reliable and consistent with expected AI behavior.",
        "MEDIUM": "The response shows moderate reliability but may need verification.",
        "LOW": "The response shows unusual patterns and should be treated with caution."
    }
    return descriptions.get(classification, "Unknown classification")

@app.route('/calculate-trust', methods=['POST'])
@rate_limit_if_default_key
def calculate_trust():
    data = request.json
    if not data or 'messages' not in data or 'output' not in data:
        return jsonify({'error': 'Invalid input'}), 400
    
    # Get API key from header or use default
    api_key = request.headers.get('X-API-Key', DEFAULT_API_KEY)
    
    if not api_key:
        return jsonify({'error': 'No API key provided'}), 401
    
    messages: List[Dict] = data['messages']
    output: str = data['output']
    
    # Calculate perplexity using the messages array and output
    perplexity = calculate_perplexity(messages, output, api_key)
    trust_result = calculate_trust_score(perplexity)
    
    return jsonify({
        'trust_score': trust_result["score"],
        'trust_classification': trust_result["classification"],
        'trust_description': trust_result["description"],
        'perplexity': perplexity,
        'using_default_key': api_key == DEFAULT_API_KEY
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)