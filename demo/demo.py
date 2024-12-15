```
🚧 WORK IN PROGRESS 🚧
This code is currently under active development and review. 
The concepts, methodologies, and technical details described here
are subject to change as we refine our approach and gather more data.
Please treat this as a draft version.
```

from openai import OpenAI
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import json
import matplotlib.pyplot as plt
from scipy import stats

from dotenv import load_dotenv
import os

load_dotenv()
openrouter_api_key = os.getenv('OPENROUTER_API_KEY')

# Initialize OpenAI client
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key,
)

# Root prompts
PROMPT_1 = "Write about the future of AI technology"
PROMPT_2 = "Write about gardening tips for beginners"

def generate_samples(prompt, n=10):
    samples = []
    for _ in range(n):
        response = client.chat.completions.create(
            model="openai/gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,  # Add some randomness
            max_tokens=100
        )
        samples.append(response.choices[0].message.content)
    return samples

def get_log_probs(text, model="openai/gpt-4"):
    """Get log probabilities for a piece of text"""
    response = client.completions.create(
        model=model,
        prompt=text,
        max_tokens=0,
        echo=True,
        logprobs=True
    )
    return response.choices[0].logprobs.token_logprobs

def calculate_confidence_score(text, reference_distribution):
    """Calculate how likely a text came from a particular distribution"""
    text_logprobs = get_log_probs(text)
    # Use KL divergence to measure similarity to reference distribution
    return stats.ks_2samp(text_logprobs, reference_distribution)

# Generate samples
print("Generating samples...")
samples_1 = generate_samples(PROMPT_1)
samples_2 = generate_samples(PROMPT_2)

# Get log probabilities for all samples
print("Calculating log probabilities...")
logprobs_1 = [get_log_probs(text) for text in samples_1]
logprobs_2 = [get_log_probs(text) for text in samples_2]

# Convert to feature vectors (using mean and std of log probs as features)
features_1 = np.array([[np.mean(lp), np.std(lp)] for lp in logprobs_1])
features_2 = np.array([[np.mean(lp), np.std(lp)] for lp in logprobs_2])

# Combine features for clustering
X = np.vstack([features_1, features_2])

# Perform clustering
kmeans = KMeans(n_clusters=2, random_state=42)
clusters = kmeans.fit_predict(X)

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(features_1[:, 0], features_1[:, 1], c='blue', label='Prompt 1')
plt.scatter(features_2[:, 0], features_2[:, 1], c='red', label='Prompt 2')
plt.xlabel('Mean Log Probability')
plt.ylabel('Std Dev of Log Probability')
plt.title('Clustering of Text Samples by Log Probabilities')
plt.legend()
plt.savefig('clustering_results.png')

# Calculate confidence scores for a test case
test_text = "AI will revolutionize how we live and work"
confidence_1 = calculate_confidence_score(test_text, np.mean(logprobs_1, axis=0))
confidence_2 = calculate_confidence_score(test_text, np.mean(logprobs_2, axis=0))

print("\nResults:")
print(f"Number of samples correctly clustered: {sum(clusters[:10] == clusters[0]) + sum(clusters[10:] == clusters[10])}")
print("\nConfidence Scores for test text:")
print(f"Probability of coming from Prompt 1: {confidence_1.pvalue:.3f}")
print(f"Probability of coming from Prompt 2: {confidence_2.pvalue:.3f}")

# Save results
results = {
    'prompt_1': PROMPT_1,
    'prompt_2': PROMPT_2,
    'samples_1': samples_1,
    'samples_2': samples_2,
    'logprobs_1': logprobs_1,
    'logprobs_2': logprobs_2,
    'clustering_accuracy': float(sum(clusters[:10] == clusters[0]) + sum(clusters[10:] == clusters[10])) / 20,
    'test_case': {
        'text': test_text,
        'confidence_prompt_1': float(confidence_1.pvalue),
        'confidence_prompt_2': float(confidence_2.pvalue)
    }
}

with open('clustering_results.json', 'w') as f:
    json.dump(results, f, indent=2)