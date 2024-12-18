from flask import Flask, request, jsonify
from flask_cors import CORS
import openai
import numpy as np
from typing import List, Dict
import os
import time
from functools import wraps
from dotenv import load_dotenv
import requests
import json
import tiktoken
import re

load_dotenv()

DEFAULT_API_KEY = os.getenv('TOGETHER_API_KEY')

app = Flask(__name__)
CORS(app, resources={
    r"/calculate-trust": {
        "origins": "*",  # Allow all origins
        "methods": ["POST", "OPTIONS"],  # Added OPTIONS for preflight requests
        "allow_headers": ["Content-Type", "X-API-Key", "Authorization"],
        "expose_headers": ["Content-Type"],
        "supports_credentials": False  # Must be False when using "*" for origins
    }
})

# Rate limiting setup
last_request_time = 0
RATE_LIMIT_SECONDS = 1

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

def check_sequence_repetition(text: str) -> bool:
    """Check for suspicious token repetition using tiktoken."""
    try:
        # Use cl100k_base encoding (same as GPT-4)
        encoding = tiktoken.get_encoding("cl100k_base")
        tokens = encoding.encode(text)
        
        # Get unique tokens and their counts
        token_set = set(tokens)
        unique_count = len(token_set)
        total_count = len(tokens)
        
        # Calculate token concentration (unique/total ratio)
        if total_count == 0:
            return False
            
        concentration = unique_count / total_count
        
        # Flag if there's high token repetition (low concentration)
        # Adjust threshold as needed (0.2 means 80% of tokens are repeats)
        return concentration < 0.2
        
    except Exception as e:
        return False  # Default to not flagging on error

def calculate_perplexity(messages: List[Dict], output: str, api_key: str) -> float:
    """Calculate perplexity for output text."""
    try:
        # First check for repetitive sequences
        if check_sequence_repetition(output):
            return float('inf')

        def normalize_text(text: str) -> str:
            """Normalize text by removing emojis and standardizing special characters"""
            # Remove emojis and other special unicode characters
            text = re.sub(r'[^\x00-\x7F]+', '', text)
            # Normalize whitespace
            text = ' '.join(text.split())
            # Normalize backslashes in shrug emoji
            text = text.replace('\\\\', '\\')
            return text.strip()

        # Clean and normalize the texts
        normalized_output = normalize_text(output)
        normalized_messages = []
        for msg in messages:
            normalized_msg = msg.copy()
            normalized_msg['content'] = normalize_text(msg['content'])
            normalized_messages.append(normalized_msg)
        
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}"
        }
        
        full_messages = normalized_messages + [{"role": "assistant", "content": normalized_output}]
        
        payload = {
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "messages": full_messages,
            "temperature": 0,
            "max_tokens": 0,
            "logprobs": 1,
            "echo": True
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response_json = response.json()
        
        if 'choices' not in response_json or not response_json['choices']:
            return float('inf')
            
        # Get logprobs and reconstruct full text
        prompt_logprobs = response_json['prompt'][0]['logprobs']
        tokens = prompt_logprobs['tokens']
        token_logprobs = prompt_logprobs['token_logprobs']
        
        # Normalize and find output position
        full_text = "".join(tokens)
        normalized_full_text = normalize_text(full_text)
        
        output_pos = normalized_full_text.find(normalized_output)
        if output_pos == -1:
            return float('inf')
        
        # Get output tokens and their logprobs
        output_text = ""
        output_token_indices = []
        normalized_output_text = ""
        
        for i, token in enumerate(tokens):
            output_text += token
            normalized_output_text = normalize_text(output_text)
            if (len(normalized_output_text) > output_pos and 
                len(normalized_output_text) <= output_pos + len(normalized_output)):
                output_token_indices.append(i)
        
        if not output_token_indices:
            return float('inf')
        
        # Get logprobs for output tokens
        output_logprobs = []
        for idx in output_token_indices:
            logprob = token_logprobs[idx]
            if logprob is not None:
                output_logprobs.append(logprob)
        
        if not output_logprobs:
            return float('inf')
        
        # Calculate simple average of logprobs
        avg_log_prob = np.mean(output_logprobs)
        perplexity = np.exp(-avg_log_prob)
        
        return perplexity
    
    except Exception as e:
        return float('inf')

def calculate_trust_score(messages: List[Dict], output: str, api_key: str) -> dict:
    """Calculate trust score using weighted perplexity."""
    # Get perplexity-based score
    perplexity = calculate_perplexity(messages, output, api_key)
    perplexity_score = np.exp(-perplexity / 100)  # Base score from perplexity

    
    # Clamp between 0 and 1
    final_score = min(max(perplexity_score, 0), 1)
    
    # Classify based on final score
    if final_score >= 0.8:
        classification = "HIGH"
    elif final_score >= 0.5:
        classification = "MEDIUM"
    else:
        classification = "LOW"
    
    return {
        "score": final_score,
        "classification": classification,
        "description": get_trust_description(classification),
        "perplexity_score": perplexity_score,
        "perplexity": perplexity
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
    
    api_key = request.headers.get('X-API-Key', DEFAULT_API_KEY)
    
    if not api_key:
        return jsonify({'error': 'No API key provided'}), 401
    
    messages: List[Dict] = data['messages']
    output: str = data['output']
    
    # Calculate trust score
    trust_result = calculate_trust_score(messages, output, api_key)
    print(f"Trust result: {trust_result}")
    
    return jsonify({
        'trust_score': trust_result["score"],
        'trust_classification': trust_result["classification"],
        'trust_description': trust_result["description"],
        'perplexity_score': trust_result["perplexity_score"],
        'perplexity': trust_result["perplexity"],
        'using_default_key': api_key == DEFAULT_API_KEY
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)