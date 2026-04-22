# Real-Time LLM Market Signal Inference Engine

## Overview

Built a real-time, low-latency AI inference system that converts live social media signals into actionable ETF trading recommendations within seconds.

The system combines:

- LLM-based structured feature extraction using AWS Bedrock
- Machine learning inference using XGBoost classifiers
- Containerized streaming deployment on AWS ECS with Fargate
- Event-driven cloud architecture using SNS, Lambda, API Gateway, DynamoDB, S3, and Aurora

It is designed as a continuously running, production-style inference system with persistent execution, low-latency signal generation, and automated signal delivery to end users.

Please use this [subscription link](https://production.d2go4b2opx7lvu.amplifyapp.com/) to sign up for testing.

<br>

## Key Capabilities

- Real-time ingestion and processing of live social media posts
- Structured LLM feature extraction into standardized numerical features
- ETF-specific ML inference using trained XGBoost models
- Secondary LLM validation and scoring for signal quality control
- Event-driven signal distribution through an email subscription system
- Continuous execution with restart handling and fault recovery

<br>

## System Architecture

![System Architecture](diagram.png)

### 1. Real-Time Inference Pipeline

The inference pipeline performs the following steps:

- Ingests live social media posts through a crawler service
- Performs preprocessing, cleaning, and deduplication
- Converts unstructured post content into structured numerical features using an LLM
- Runs ETF-specific ML inference for signal prediction
- Applies a secondary LLM-based validation and scoring layer
- Publishes qualified signals to downstream delivery systems via SNS

### 2. Containerized Inference Deployment

The inference service is containerized and deployed in AWS:

- Docker image stored in Amazon ECR
- Deployed on AWS ECS with Fargate for serverless container execution
- Supports persistent operation and production-style deployment
- Models are preloaded from S3 at container startup to reduce runtime latency

### 3. Cloud-Based Subscription System

Users can subscribe to receive real-time ETF signals.

Components:

- Frontend: AWS Amplify
- API Layer: Amazon API Gateway
- Backend: AWS Lambda
- Messaging: Amazon SNS

Signal delivery flow:

`User -> Amplify -> API Gateway -> Lambda -> SNS -> Email Notification`

SNS acts as the central event bus connecting signal generation with subscriber delivery.

### 4. Storage Layer

The system uses multiple storage services for different purposes:

- DynamoDB (Posts): tracks processed posts for deduplication
- DynamoDB (Signals): stores processed and published signals
- S3: stores model artifacts and training data
- Aurora: supports validation, threshold, and scoring queries

### 5. Model Training Pipeline

The training pipeline is built using historical post data and ETF market data.

It includes:

- Hyperparameter search
- Multi-run stability evaluation
- Threshold optimization
- High-confidence subset filtering

Trained models are stored in S3 and loaded into the live inference pipeline during deployment.

### 6. Low-Latency System Design

The system is designed to reduce inference latency:

- Models are preloaded into memory to avoid runtime I/O
- Concurrent LLM invocation is used to improve throughput
- Event-driven architecture reduces blocking dependencies
- Signals are designed to be generated within seconds of post ingestion

### 7. LLM Feature Engineering

Instead of relying on raw embeddings, the system uses structured LLM outputs as the core feature representation.

The LLM:

- Converts text into a multi-dimensional structured feature space
- Uses a strict ordinal scoring schema
- Captures geopolitical, economic, and market-relevant signals

Benefits of this design:

- Improved model stability
- Higher interpretability
- Better downstream ML performance

### 8. ML Model Retraining Pipeline

The system also includes an automated ML model retraining pipeline designed to refresh deployed ETF models using newly accumulated post and market data.

The retraining workflow performs the following steps:

- Loads historical structured LLM output and aligned ETF market data
- Rebuilds the supervised training dataset for each covered ETF
- Runs large-scale XGBoost optimization across ETF-specific feature combinations
- Evaluates model robustness across repeated random-state runs
- Applies automated LLM-based model selection to identify the strongest candidates
- Saves selected models and updated performance outputs for downstream deployment

To support larger-scale retraining workloads, the retraining job is containerized separately and deployed on AWS ECS with Fargate. This design allows retraining to run independently from the live inference service while sharing the same cloud-native infrastructure stack.

The retraining system is built to support:

- Symbol-level training control through ECS runtime configuration
- Repeatable large-scale optimization workflows
- Separation between live inference and offline model refresh
- Production-style model lifecycle management

### 9. Performance Evaluation

The system includes a post-prediction evaluation layer that measures how published ETF signals perform against future market movement.

Predictions are evaluated using forward VWAP-based return windows after signal generation. This allows the system to measure whether each published buy or sell signal aligned with subsequent price direction over a defined horizon.

The evaluation pipeline supports:

- Forward return calculation across multiple horizons
- Directional accuracy measurement for published signals
- Signal-aligned return analysis for buy and sell predictions
- Symbol-level performance breakdown
- Buy-side versus sell-side performance comparison
- Confidence-filtered performance review using LLM scoring outputs

These evaluations are used both for internal monitoring and for periodic reporting. The framework supports summary generation at multiple levels, including:

- Overall signal performance
- Performance by symbol
- Performance for buy-only signals
- Performance for sell-only signals
- Performance for high-reasonableness signals

This evaluation layer provides an important feedback loop between live deployment, signal quality control, and future retraining decisions.

### 10. Daily News Feed

The daily news feed pipeline builds a structured article corpus for topic matching, RAG retrieval, and topic memory generation.

The workflow performs the following steps:

- Collects English-language U.S. news from TheNewsAPI across configurable date and intraday ET windows
- Converts ET time windows into UTC API query windows
- Applies category, language, locale, and optional domain filters
- Extracts full article text from source URLs
- Removes boilerplate content and filters weak extracted text
- Stores deduplicated daily news in S3 as date-partitioned Parquet files

### 11. Topic Memory System

The topic memory system creates a compact news memory layer by summarizing daily article coverage at the topic level.

The workflow performs the following steps:

- Loads news-topic matching records from Aurora and partitions them by topic
- Serializes each topic-specific article subset into CSV context for Bedrock document-based summarization
- Generates structured topic memory with `topic`, `processing_date`, `overall_topic_summary`, `seven_day_summary`, and `three_day_summary`
- Persists topic summary outputs into Aurora as augmented topic memory for future post querying and background-context retrieval

### 12. RAG System

The RAG system converts topic-matched news articles into retrieval-ready embedding vectors stored in Aurora PostgreSQL with pgvector.

The workflow performs the following steps:

- Consolidates long-form UUID-topic mappings into UUID-level article records with topic-list metadata
- Joins article-level topic metadata with extracted full text and source information
- Applies semantic sentence-aware chunking with overlap, paragraph/transition boundary detection, and stop-reason tracking
- Prepends retrieval metadata headers to each chunk before embedding
- Generates 1024-dimensional embedding vectors with Amazon Titan Text Embeddings V2 and loads chunk records into Aurora pgvector for similarity search

<br>

## Model Performance

The deployed ETF-specific models are built using XGBoost and evaluated as binary directional classifiers for buy and sell prediction. Model performance is measured on out-of-sample validation data to reflect generalization rather than in-sample fit.

Across the covered ETF universe, the models consistently demonstrate statistically meaningful directional predictive power, with accuracy levels above the random baseline of `0.50`. In the context of financial time series, even moderate improvements over random performance can represent meaningful predictive edge.

### Accuracy Benchmark

Because market prices are noisy and partially efficient, directional prediction accuracy should be interpreted using domain-specific standards rather than conventional classification expectations.

| Accuracy Range | Interpretation |
| --- | --- |
| < 0.52 | Noise / no meaningful signal |
| 0.52 – 0.55 | Weak edge |
| 0.55 – 0.60 | Consistent signal |
| 0.60 – 0.65 | Strong directional signal |
| 0.65+ | High-confidence signal (rare in real markets) |

### Model Accuracy

| Symbol | Model Accuracy - Manual Selected | Model Accuracy - LLM Selected | Model Coverage |
| :---: | :---: | :---: | :---: |
| DIA | 0.623 | 0.625 | 0.649 |
| HYG | 0.672 | 0.678 | 0.776 |
| NVDU | 0.592 | 0.585 | 0.492 |
| QQQ | 0.616 | 0.624 | 0.322 |
| SOXX | 0.605 | 0.608 | 0.476 |
| SPY | 0.616 | 0.616 | 0.420 |
| TLT | 0.583 | 0.582 | 0.382 |
| TSLL | 0.580 | 0.585 | 0.307 |
| UCO | 0.616 | 0.611 | 0.466 |
| UGL | 0.596 | 0.581 | 0.380 |
| VXX | 0.621 | 0.623 | 0.397 |
| XLE | 0.562 | 0.566 | 0.286 |
| XLF | 0.617 | 0.609 | 0.499 |

### Key Observations

- Most deployed models fall within the `0.58 – 0.67` range, placing the system broadly in the usable-to-strong signal category.
- Several core ETFs including `DIA`, `QQQ`, `SPY`, `UCO`, `VXX`, and `XLF` exceed `0.60`, indicating strong directional performance across multiple asset classes.
- `HYG` currently shows the highest model accuracy at `0.672008`, suggesting particularly strong predictive capture in credit-sensitive market behavior.
- Performance is relatively stable across equities, commodities, volatility, and macro proxy ETFs, indicating good cross-asset robustness.

### Practical Interpretation

In real financial markets, achieving directional accuracy above `0.60` is already meaningful due to high noise, regime shifts, and relatively low signal-to-noise ratio. For this reason, the system is designed not only to optimize raw model accuracy, but also to improve effective signal quality through additional inference controls.

These include:

- LLM-based validation and signal filtering
- Confidence-based thresholding
- Real-time inference and decision control

Together, these layers are intended to improve real-world signal precision beyond standalone model metrics and support a more robust production inference pipeline.

<br>

## Roadmap

Planned future improvements include:

- Real-time news integration through RAG or agent-based workflows
- Multi-modal signal ingestion across text, image, and video
- Further latency optimization and scaling improvements

<br>

## Why This Project Matters

This project demonstrates:

- End-to-end ML system design beyond just model training
- Real-time inference under latency constraints
- Hybrid LLM + ML architecture
- Cloud-native production deployment on AWS
- Event-driven system design with a user-facing delivery product
