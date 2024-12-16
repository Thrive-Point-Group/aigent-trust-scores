# Plex: Simple and Transparent AI Bot Verification

Welcome to **Plex**, a lightweight and powerful tool for verifying the authenticity of AI-generated messages in the cryptocurrency ecosystem. By analyzing statistical patterns in AI outputs, Plex helps distinguish genuine bot-generated messages from potentially manipulated or malicious ones.

---

## Why Use Plex?

1. **Build Trust:** Ensure AI-generated messages are authentic and reliable.
2. **Enhance Transparency:** Promote trust in AI communications across the crypto landscape.
3. **Strengthen Security:** Protect against manipulation by validating AI outputs with robust statistical methods.

---

## What Does Plex Do?

Plex evaluates AI-generated messages by:
- Analyzing **log probabilities** to assess the model's confidence in its predictions.
- Calculating **perplexity**, a metric for determining how predictable the output is.
- Generating a **trust score** to quantify authenticity.
- Comparing outputs against trusted datasets to flag anomalies.

---

## Quick Links

- **[Litepaper](WHITEPAPER.md):** Learn the core methodology and technical principles behind Plex.
- **[API Documentation](docs/):** Get started with our public API for integrating Plex into your tools.
- **Public API URL:** [https://plex.higherrrrrrr.fun](https://plex.higherrrrrrr.fun)

---

## How It Works (Simplified)

1. **Submit Context and Output:** Provide a sequence of input messages and the AI-generated output.
2. **Statistical Analysis:** Plex evaluates the log probabilities of each token and calculates the perplexity of the message.
3. **Generate Trust Score:** A trust score is computed, indicating the likelihood that the message originated from a legitimate AI model.

---

## Features

- **Model Agnostic:** Compatible with GPT-based models and any system that provides token-level log probabilities.
- **Simple API Integration:** Easily integrate Plex with your crypto or AI applications.
- **Robust Analysis:** Incorporates advanced statistical tests for reliable verification.
- **Lightweight and Fast:** Minimal overhead for real-time validation needs.

---

## Example Workflow

1. **Send a Request:** Provide the AI context (message array) and the generated output.
2. **Get Results:** Receive perplexity and trust scores as JSON responses.
3. **Evaluate Authenticity:** Use the trust score to decide if the message is trustworthy.

---

## API Overview

### Endpoint: `POST /calculate-trust`

#### Request Body:
```json
{
  "messages": [
    {"role": "user", "content": "What is Bitcoin?"},
    {"role": "assistant", "content": "Bitcoin is a decentralized digital currency."}
  ],
  "output": "It operates without a central authority."
}
```

#### Response:
```json
{
  "trust_score": 0.87,
  "perplexity": 15.32,
  "using_default_key": false,
  "trust_classification": "HIGH",
  "trust_description": "The response appears highly reliable and consistent with expected AI behavior."
}
```

---

## Get Started Today

Use Plex to bring trust, transparency, and security to your AI communications. Explore the **[API Documentation](docs/)** or dive into the **[Litepaper](WHITEPAPER.md)** to learn more.