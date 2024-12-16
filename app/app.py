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

def calculate_perplexity(messages: List[Dict], output: str, api_key: str) -> float:
    """Calculate perplexity for a given set of messages and output using token probabilities."""
    try:
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}"
        }
        
        full_messages = messages + [{"role": "assistant", "content": output}]
        
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
        
        # Reconstruct the full text and track token positions
        full_text = ""
        token_positions = []  # List of (start, end) positions for each token
        
        for token in tokens:
            start = len(full_text)
            full_text += token
            end = len(full_text)
            token_positions.append((start, end))
        
        # Find the output string position
        output_pos = full_text.find(output)
        if output_pos == -1:
            return float('inf')
        
        # Find which tokens overlap with our output string
        output_end = output_pos + len(output)
        output_token_indices = []
        
        for i, (start, end) in enumerate(token_positions):
            # Check if this token overlaps with our output string
            if end > output_pos and start < output_end:
                output_token_indices.append(i)
        
        # Get logprobs for those tokens
        output_logprobs = [token_logprobs[i] for i in output_token_indices]
        valid_logprobs = [lp for lp in output_logprobs if lp is not None]
        
        if not valid_logprobs:
            return float('inf')
            
        avg_log_prob = np.mean(valid_logprobs)
        perplexity = np.exp(-avg_log_prob)
        
        return perplexity
    
    except Exception as e:
        return float('inf')

def get_expected_output(messages: List[Dict], api_key: str) -> str:
    """Get expected output by querying the model with the conversation history."""
    try:
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}"
        }
        
        # Use all messages except the last assistant message
        context_messages = [msg for msg in messages if msg["role"] != "assistant"]
        
        payload = {
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "messages": context_messages,
            "temperature": 0,
            "max_tokens": 1000
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response_json = response.json()
        
        if 'choices' not in response_json or not response_json['choices']:
            return ""
            
        expected_output = response_json['choices'][0]['message']['content']
        return expected_output
        
    except Exception as e:
        print(f"Error getting expected output: {e}")
        return ""

def calculate_embedding_similarity(messages: List[Dict], output: str, api_key: str) -> float:
    """Calculate similarity between expected and actual output using LLM judgment."""
    try:
        url = "https://api.together.xyz/v1/chat/completions"
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            "authorization": f"Bearer {api_key}"
        }
        
        # Get model's expected output for this context
        expected_output = get_expected_output(messages, api_key)
        
        print("\n=== Similarity Judgment ===")
        print(f"Expected Output: {repr(expected_output)}")
        print(f"Actual Output: {repr(output)}")
        
        # Create prompt for the LLM to judge similarity
        judge_prompt = [
            {
                "role": "system",
                "content": """You are a judge evaluating the semantic similarity between two texts.
                Compare them for meaning, factual consistency, and intent.
                Respond in JSON format with the following fields:
                - similarity_score: float between 0 and 1
                - reasoning: brief explanation of your score
                - key_differences: list of main differences (if any)
                - factual_consistency: boolean indicating if the facts align
                """
            },
            {
                "role": "user",
                "content": f"""Compare these two texts for similarity:

Text 1 (Expected):
{expected_output}

Text 2 (Actual):
{output}

Provide your judgment in JSON format."""
            }
        ]
        
        # Get LLM judgment
        payload = {
            "model": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
            "messages": judge_prompt,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "max_tokens": 1000
        }
        
        response = requests.post(url, json=payload, headers=headers)
        response_json = response.json()
        print(f"Judge Response: {response_json}")
        
        if 'choices' not in response_json or not response_json['choices']:
            return 0.0
            
        # Parse judgment
        judgment = json.loads(response_json['choices'][0]['message']['content'])
        similarity_score = float(judgment.get('similarity_score', 0.0))
        
        print(f"Similarity Score: {similarity_score}")
        print(f"Reasoning: {judgment.get('reasoning', 'No reasoning provided')}")
        print("=====================================\n")
        
        return similarity_score
    
    except Exception as e:
        print(f"Error calculating similarity: {e}")
        return 0.0

def calculate_hybrid_score(messages: List[Dict], output: str, api_key: str) -> dict:
    """Calculate a hybrid trust score using perplexity and penalizing for low embedding similarity."""
    # Get perplexity-based score
    perplexity = calculate_perplexity(messages, output, api_key)
    perplexity_score = np.exp(-perplexity / 100)  # Base score from perplexity
    
    # Get embedding similarity score
    similarity_score = calculate_embedding_similarity(messages, output, api_key)
    
    # Calculate penalty using a modified sigmoid curve
    # Shifted and scaled to:
    # - Almost no penalty (>0.98) for similarity above 0.6
    # - Very steep drop-off below 0.5
    # - Near-zero (<0.02) for very low similarity
    steepness = 15  # Controls how sharp the transition is
    midpoint = 0.2  # Shifts where the steep drop-off occurs
    penalty = 1 / (1 + np.exp(-steepness * (similarity_score - midpoint)))
    
    # Apply penalty to perplexity score
    final_score = perplexity_score * penalty
    
    # Clamp between 0 and 1
    final_score = min(max(final_score, 0), 1)
    
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
        "similarity_score": similarity_score,
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
    
    # Calculate hybrid trust score
    trust_result = calculate_hybrid_score(messages, output, api_key)
    
    return jsonify({
        'trust_score': trust_result["score"],
        'trust_classification': trust_result["classification"],
        'trust_description': trust_result["description"],
        'perplexity_score': trust_result["perplexity_score"],
        'similarity_score': trust_result["similarity_score"],
        'perplexity': trust_result["perplexity"],
        'using_default_key': api_key == DEFAULT_API_KEY
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)