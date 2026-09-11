# Citations & Attributions

This document acknowledges datasets, algorithms, model architectures, and libraries utilized within the Hiver AI Twitter Support Agent repository.

## 1. Dataset
- **Kaggle Customer Support on Twitter Dataset**
  - Source: [Kaggle: thoughtvector/customer-support-on-twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
  - Authors: Thoughtvector / Kaggle Community
  - Description: Corpus of ~2.8M customer support tweets and replies from top corporate brands on Twitter, capturing inbound customer inquiries and authentic company resolution threads.
  - License: CC-BY-NC-SA 4.0

## 2. Models & Architectures
- **Sentence-Transformers (all-MiniLM-L6-v2)**
  - Source: [Hugging Face sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  - Reference: Reimers, N., & Gurevych, I. (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. EMNLP 2019.
  - Usage: Local, CPU-efficient dense embeddings (384-dimensional) for semantic vector indexing and retrieval over historical resolved threads.

- **Large Language Models (LLM Inference)**
  - Ollama: Local inference runner supporting open-weights instruction models (e.g. Qwen 2.5, Llama 3).
  - OpenAI: gpt-4o-mini API integration for zero/few-shot intent classification and RAG response drafting.
  - Anthropic: claude-3-5-haiku-20241022 API integration.

## 3. Machine Learning & Metric Evaluation
- **TF-IDF Vectorizer & Logistic Regression Baseline**
  - Scikit-Learn: Pedregosa et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, pp. 2825-2830.
  - Unigram/bigram sublinear term frequency weighting paired with L2-regularized multinomial logistic regression.

- **Evaluation Metrics & Inter-Annotator Agreement**
  - Quadratic-Weighted Cohen's Kappa: Cohen, J. (1968). *Weighted kappa: nominal scale agreement with provision for scaled disagreement or partial credit*. Psychological Bulletin, 70(4), 213-220.
  - Macro-F1 and Precision/Recall: Standard multiclass classification and binary risk-weighted escalation metrics implemented via sklearn.metrics.
