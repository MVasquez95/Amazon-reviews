Amazon Reviews Sentiment Analysis

Author: Miguel Vásquez
Date: September 2025

Project Overview

This project performs sentiment analysis on Amazon product reviews. The goal is to classify reviews as positive or negative using both classical machine learning models (Logistic Regression, SVM) and transformer-based models (DistilBERT, optionally BERT-base).

The workflow includes:

Exploratory Data Analysis (EDA): Investigating class distribution, review lengths, common words, n-grams, and sentiment-specific vocabulary.

Feature Engineering & Preprocessing:

TF-IDF vectorization for traditional ML models.

Tokenization and sequence preparation for transformer models.

Model Training & Evaluation: Comparing models in terms of accuracy and training time.

Conclusions & Recommendations: Selecting the best approach for portfolio demonstration and practical applications.

File Structure
.
├── results/                  # Model outputs and evaluation results (large, not included)
├── amazon_reviews_clean.csv  # Cleaned dataset (large, not included)
├── amazon_reviews_subset.csv # Subset of data for testing/training (large, not included)
├── notebooks.ipynb           # Jupyter notebooks for EDA, modeling, and evaluation
└── README.md                 # Project documentation


Note: The results/, amazon_reviews_clean.csv, and amazon_reviews_subset.csv files are excluded from Git via .gitignore due to their size.
To reproduce these files, follow the preprocessing steps described in the notebooks or scripts.

Data

The dataset contains Amazon product reviews, spanning multiple years.

Each review includes the text, a rating, and metadata such as product and user IDs.

A subset was used for faster experimentation and portfolio demonstration.

How to Reproduce

Download the raw Amazon reviews dataset (publicly available or provided source).

Run the preprocessing notebook/script to clean the data and generate amazon_reviews_clean.csv.

Optionally, create a smaller subset amazon_reviews_subset.csv for quicker training and testing.

Execute the modeling notebook/script to train models and generate results.

Dependencies

Python >= 3.10

pandas, numpy, matplotlib, seaborn

scikit-learn

transformers (Hugging Face)

datasets (Hugging Face)

torch (PyTorch)

Results

Classical models (Logistic Regression, SVM) achieve ~86% test accuracy.

DistilBERT achieves ~93% test accuracy, showing the advantage of transformer-based models for NLP tasks.

Training times vary significantly between classical models (seconds) and transformers (~76 minutes for 30k samples with DistilBERT).

License

This project is for educational and portfolio purposes.