Sentiment Analysis Using DistilBERT
Overview
This project implements a sentiment analysis system using DistilBERT, a lighter, faster version of BERT, for classifying movie reviews as positive or negative. The IMDB movie review dataset, accessed via Hugging Face Datasets, serves as the primary dataset for training and evaluation.

Table of Contents
Overview
Dataset# Sentiment Analysis Using DistilBERT

## Overview
This project implements a sentiment analysis system using **DistilBERT**, a lighter, faster version of BERT, for classifying movie reviews as positive or negative. The IMDB movie review dataset, accessed via **Hugging Face Datasets**, serves as the primary dataset for training and evaluation.

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Preprocessing](#preprocessing)
- [Training and Evaluation](#training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [References](#references)

## Dataset
The dataset used for this project is the **IMDB movie review dataset**, which is available through the **Hugging Face Datasets library**. It contains 50,000 movie reviews, evenly split between positive and negative sentiments. The dataset is widely used in NLP for sentiment classification tasks.

## Model Architecture
The model leverages **DistilBERT**, which is a streamlined version of BERT, maintaining about 97% of BERT’s performance while being more efficient:
- **Layers**: 6 transformer layers.
- **Hidden Units**: 768 units per layer.
- **Attention Heads**: 12 attention heads.

## Preprocessing
Minimal preprocessing was necessary due to the dataset's pre-cleaned nature. The steps included:
- **Tokenization**: Using WordPiece tokenizer from the DistilBERT architecture.
- **Padding/Truncation**: Input sequences were standardized to a maximum length of 128 tokens.
- **Label Encoding**: Sentiment labels were already encoded as 1 (positive) and 0 (negative).

## Training and Evaluation
The model was trained over 25 epochs with performance measured using metrics such as **accuracy**, **precision**, **recall**, **f1-score**, and **ROC AUC**. Hyperparameters like learning rate and batch size were tuned for optimal performance.

## Results
The final evaluation yielded:
- **Training Accuracy**: 83.25%
- **Validation Accuracy**: 82%
- **Classification Report**:

  | Sentiment | Precision | Recall | F1-score | Support |
  |-----------|-----------|--------|----------|---------|
  | Negative  | 0.85      | 0.79   | 0.82     | 52      |
  | Positive  | 0.79      | 0.85   | 0.82     | 48      |
  | **Overall** | **0.82** | **0.82** | **0.82** | **100** |

- **ROC AUC Score**: 0.89

## Conclusion
The sentiment analysis model built with DistilBERT achieved strong performance, demonstrating its capability to provide efficient and effective natural language understanding in resource-constrained environments.

## References
- Hugging Face Datasets: [link]
- DistilBERT Paper: [link]
