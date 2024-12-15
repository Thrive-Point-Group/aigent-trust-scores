# TrustScore: Simple and Transparent AI Bot Verification

Welcome to **TrustScore**, a tool designed to help verify whether AI-generated messages in the cryptocurrency ecosystem are authentic and trustworthy. This project focuses on analyzing how likely a sequence of messages (input context) and an output message together originate from a legitimate AI model by looking at statistical patterns in the model's outputs.

## Why Use TrustScore?
1. **Ensure Trust:** Differentiate between real AI-generated messages and potential fakes.
2. **Improve Transparency:** Encourage all participants in the ecosystem to share input/output pairs for better analysis.
3. **Promote Collaboration:** Build a safer and more reliable cryptocurrency ecosystem by validating bot communications.

## What Does TrustScore Do?
TrustScore evaluates AI message arrays by:
- Extracting **log probabilities** from model outputs (basically how confident the AI was about each word in the context and output).
- Comparing these patterns against trusted datasets.
- Scoring messages based on how closely they match authentic ones.

## Quick Links
- **[Whitepaper](WHITEPAPER.md):** Dive deep into the math and methodology behind TrustScore.
- **[Demo Code](demo/):** Check out examples to see how TrustScore works in action.

## How It Works (Simplified)
1. **Input a Message Array:** You provide a sequence of messages (context) and the final output message from an AI bot.
2. **Run Analysis:** TrustScore looks at the likelihood of each word in the array and compares it with patterns in trusted bot outputs.
3. **Get a Score:** The system tells you how much it trusts the message array, helping you flag any suspicious activity.

## Features
- **Compatible with Any Model:** Works with GPT, other transformers, or any model that outputs token probabilities.
- **Lightweight API:** Simple to integrate into your existing crypto tools.
- **Statistical Robustness:** Uses advanced tests like the Kolmogorov-Smirnov (KS) test to ensure reliability.

