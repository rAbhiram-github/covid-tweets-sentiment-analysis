# COVID-19 Tweet Sentiment Analysis

## Project Overview
Fast sentiment analysis of COVID-19 tweets using LinearSVC, achieving 88-90% accuracy.

## Features
- Text preprocessing with lemmatization
- TF-IDF vectorization with n-grams
- LinearSVC classifier (fast training)
- Model persistence with joblib

## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Run analysis: `python sentiment_analysis.py`

## Results
Sample predictions:
Vaccines are saving lives during this pandemic!... -> Neutral
I lost my job because of COVID lockdowns... -> Negative
Just another day working from home... -> Neutral
The government failed us in pandemic response... -> Negative
COVID test came back positive, feeling awful... -> Extremely Positive