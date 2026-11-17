# AI-POWERED NEWS CHECKER
### A Three-Layer AI-Based Fake News Detection and Verification System

## Project Overview

AI-Powered News Checker is an intelligent news verification system designed to analyse online news articles and identify potentially fake or genuine information.

The project combines a fine-tuned DistilBERT NLP model, real-time web research, and a locally running Qwen3-8B language model to perform multi-layer news verification.

The DistilBERT model is described as having been trained on a dataset of over 60,000 news articles. The system then gathers related information from online sources and passes the collected evidence to an AI-based judging model for a final assessment.

The objective is to make online news verification more comprehensive by combining machine learning, web research, and AI-based evidence evaluation.

## Problem Statement

The rapid spread of misinformation and misleading news across the internet makes it increasingly difficult to distinguish genuine information from fake news.

A single machine learning prediction may not provide sufficient context to verify the claims made in a news article.

This project addresses this challenge through a three-layer verification approach. It first analyses the article using a trained NLP model, then searches for related information across the internet, and finally uses a large language model to evaluate the collected evidence.

## Technology Stack

- **Python:** Core programming language.
- **Flask:** Backend framework and web application interface.
- **PyTorch:** Deep learning model execution.
- **Hugging Face Transformers:** Loading and running the DistilBERT classification model.
- **DistilBERT:** Fine-tuned NLP model for initial fake and genuine news classification.
- **Qwen3-8B (Q6_K):** Local language model used for final evidence evaluation.
- **llama.cpp:** Local inference server for running the Qwen model.
- **BeautifulSoup:** Extracting article text from webpages.
- **Requests:** Fetching webpage content and online information.
- **DuckDuckGo Search:** Searching for related news articles and web sources.
- **Google News RSS:** Alternative news-search mechanism.

## Three-Layer System Architecture

### Layer 1: NLP-Based Article Classification

The first layer uses a fine-tuned DistilBERT Transformer model to analyse the submitted news article.

The model processes the article's textual content and generates an initial classification of whether the news appears FAKE or REAL, along with a confidence score.

The system supports article extraction directly from a webpage URL as well as manually pasted article text.

### Layer 2: Internet-Based News Research

The second layer performs additional research using online news and web search.

It uses DuckDuckGo News Search and Web Search to discover related articles and supporting information. Google News RSS acts as a fallback when the primary search mechanism fails.

The system collects relevant search results, filters duplicate titles and domains, and attempts to retrieve the full text of selected articles.

This layer provides additional information and supporting evidence for the final evaluation.

### Layer 3: AI-Based Final Evaluation Using Qwen3-8B

The third layer acts as the final AI-based fact-checking judge.

The collected article content and information from the web research layer are passed to the locally running Qwen3-8B model.

The model evaluates the available evidence and examines individual claims by classifying them as:

- **SUPPORTED:** The claim is supported by the available evidence.
- **CONTRADICTED:** The claim is contradicted by the available evidence.
- **UNVERIFIED:** The available sources do not establish the claim.

Based on the evidence, the model generates a final verdict:

- **REAL:** The major claims are supported by the researched sources.
- **FAKE:** A major claim is contradicted by the evidence.
- **MIXED:** Important claims remain unverified or the evidence is conflicting.

The final output also includes a confidence score and a short reasoning summary.

## How It Works

The system follows this verification workflow:

- The user submits a news article through a webpage URL or by pasting the article text.
- The system extracts and processes the article content.
- The DistilBERT model performs the initial news classification.
- The web research layer searches for related news articles and supporting information.
- Selected articles are retrieved and their content is collected.
- The Qwen3-8B model evaluates the submitted article against the collected evidence.
- The system displays the final verdict, confidence score, reasoning, claim-level assessments, and researched sources.

If web sources are unavailable or the AI judge cannot be loaded, the application can fall back to displaying the initial DistilBERT classification.

## Installation and Configuration

### Prerequisites

- Python
- PyTorch
- Hugging Face Transformers
- Flask
- llama.cpp
- A locally saved DistilBERT classification model
- Qwen3-8B GGUF model file

### Clone the Repository

```bash
git clone https://github.com/susmitacreates/AI-POWERED-NEWS-CHECKER.git
```

### Navigate to the Project Directory

```bash
cd AI-POWERED-NEWS-CHECKER
```

### Install Python Dependencies

Install the Python packages required by the application, including:

```bash
pip install flask torch requests beautifulsoup4 transformers ddgs
```

### Configure the Models

Update the model and directory paths in `finalfactchecker.py` to match your local environment.

Configure the DistilBERT model directory and the Qwen3-8B GGUF model location.

Ensure that the llama.cpp server executable is available at the configured path.

### Run the Application

```bash
python finalfactchecker.py
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

## Demo

A demonstration video is available in this GitHub repository.

It showcases the news verification interface, the three-layer processing workflow, and the final AI-generated verification results.

## Future Improvements

- Expand the news dataset and improve model evaluation.
- Enhance the accuracy of initial news classification.
- Improve online source discovery and evidence collection.
- Introduce more comprehensive claim-level verification.
- Improve the explainability and reliability of AI-generated verdicts.
- Make the application easier to configure and deploy across different systems.

## Project Repository

[AI-Powered News Checker – GitHub](https://github.com/susmitacreates/AI-POWERED-NEWS-CHECKER)

**Developed by:** Susmita Ganguly
