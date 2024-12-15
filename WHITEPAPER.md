# TrustScore: Authenticity Verification for AI Bots in Crypto

## Overview

The TrustScore method is a systematic framework designed to verify the authenticity of AI-generated messages in the cryptocurrency ecosystem. By leveraging statistical analysis of log probabilities derived from AI models, TrustScore offers a robust mechanism to detect anomalies and identify trustworthy content. The method extracts token-level log probabilities from AI-generated messages, compares these against reference distributions, and uses statistical tests such as the Kolmogorov-Smirnov test to quantify trustworthiness. This process enables stakeholders to differentiate between authentic AI outputs and potentially malicious or manipulated content.

This method is essential for fostering transparency and trust in the rapidly evolving AI-powered crypto landscape. To achieve widespread effectiveness, TrustScore relies on the collective participation of AI agents in the ecosystem. By encouraging all agents to share their input/output pairs and model specifics, the system can build comprehensive reference distributions for validation. Such collaboration would ensure robust cross-model compatibility and improve the security and reliability of AI-driven communications across the cryptocurrency ecosystem.

## Features

- **Log Probability Analysis**: Evaluates token-level confidence in AI-generated messages.
- **Confidence Scores**: Quantifies trustworthiness of messages using statistical tests.

## Log Probability Analysis: Technical Details

### Definition

The log probability of a token in a sequence is defined as:

Where:

- : The token at position .
- : The preceding tokens in the sequence.
- : The conditional probability of  given the preceding context.

Log probabilities quantify the model’s confidence for each token in a sequence, with lower values indicating less certainty.

### Theoretical Foundations

#### Conditional Probability

For any sequence of tokens , the probability of the sequence can be decomposed as:

The logarithmic transformation simplifies the product into a summation:

#### Measuring Confidence

The log probability for a single token, , is directly obtained from the language model. By aggregating log probabilities across tokens, we derive useful statistics:

- **Mean Log Probability**:
- **Standard Deviation**:

These metrics form the basis for comparing messages by characterizing the distribution of log probabilities.

### Generality Across Models

One key strength of the log probability method is its adaptability to various language models. Although different models (e.g., GPT-3, GPT-4, and other transformer-based architectures) may differ in specific architectures or training data, the underlying statistical characteristics of their outputs often exhibit consistent patterns when applied to similar datasets. As a result:

- **Model Independence**: TrustScore can be used with any AI model capable of providing token-level log probabilities.
- **Cross-Model Validation**: The same reference distributions can be employed across models with minor calibration.

#### Why Does This Work?

1. **Shared Dataset Characteristics**: Most language models are trained on broadly similar corpora, leading to analogous token probability distributions.
2. **Stable Linguistic Patterns**: Log probabilities capture fundamental aspects of linguistic predictability, which are consistent across high-quality models.
3. **Scalable Calibration**: Reference distributions can be updated incrementally to accommodate slight variations in model behavior.

### Statistical Comparison

To evaluate authenticity, the log probabilities of a test message are compared against a reference distribution:

#### Kolmogorov-Smirnov (KS) Test

The KS test compares the empirical cumulative distribution functions (CDFs) of two datasets. For log probabilities:

Where  and  represent the CDFs of the reference and test distributions, respectively. The resulting p-value indicates similarity:

- **High p-value**: Strong alignment with the reference distribution (authentic).
- **Low p-value**: Poor alignment, potential anomaly.

#### Distribution Analysis

Beyond p-values, inspecting the shape of distributions provides additional insights:

- Skewed distributions may indicate unusual behavior.
- Tight clusters around the mean suggest consistency with training data.

### Implementation Details

#### Tokenization and Log Probability Extraction

The tokenization and log probability extraction process involves converting input text into tokens and retrieving the associated log probabilities. Given a message :

1. **Tokenize**:
2. **Extract Log Probabilities**:
   For each token:

#### Statistical Feature Extraction

From the log probabilities of a message:

1. Compute the **mean log probability** ().
2. Compute the **standard deviation** ().
3. Optionally, extract higher-order moments like skewness and kurtosis for detailed analysis.

#### Comparison with Reference Distributions

Given a reference distribution , evaluate the similarity of a test message  using:

1. **Kolmogorov-Smirnov Test**:
   Compare the CDF of  with .
2. **Visual Analysis**:
   Generate histograms and kernel density estimates for alignment visualization.

#### Practical Implementation

**Code Snippets:**

```python
from trustscore import tokenize, get_log_probs

# Tokenization and Log Probability Extraction
message = "AI bots enhance trading."
tokens = tokenize(message)
log_probs = get_log_probs(message)

# Statistical Feature Extraction
import numpy as np
mean_log_prob = np.mean(log_probs)
std_log_prob = np.std(log_probs)

# KS Test for Authenticity
from scipy.stats import ks_2samp
ks_stat, p_value = ks_2samp(log_probs, reference_distribution)
print("KS Statistic:", ks_stat, "P-Value:", p_value)
```

### Application of Results

- **High p-value**: Message is consistent with reference data and likely authentic.
- **Low p-value**: Message deviates significantly, warranting further scrutiny.

### Example Workflow

1. **Reference Distribution Creation**:
   Collect log probabilities from trusted AI-generated messages.
2. **Incoming Message Analysis**:
   Tokenize and extract log probabilities for the new message.
3. **Statistical Comparison**:
   Use the KS test and other metrics to evaluate the message against the reference.
4. **Decision Making**:
   Classify the message as authentic or suspect based on thresholds.

## Limitations

- **Model Dependency**: TrustScore relies on the language model’s log probability behavior. Updates to the model may necessitate recalibration.
- **Threshold Sensitivity**: Improper thresholds can lead to false positives or negatives.
- **Privacy Concerns**: Access to message content requires strict data protection protocols.

## References

- [OpenAI API Documentation](https://beta.openai.com/docs/)
- [Kolmogorov-Smirnov Test](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)

---

*TrustScore provides a robust framework for enhancing trust in AI communications within the cryptocurrency ecosystem.*

