# Amazon Reviews Sentiment Analysis

**Author:** Miguel Vásquez  
**Date:** September 2025  

---

## 📖 Project Overview

This project performs **sentiment analysis** on Amazon product reviews. The goal is to classify reviews as **positive** or **negative** using both:

- **Classical ML models:** Logistic Regression, SVM  
- **Transformer-based models:** DistilBERT (and optionally BERT-base for future experiments)  

The workflow includes:

1. **Exploratory Data Analysis (EDA):**  
   - Class distribution  
   - Review lengths  
   - Common words and n-grams  
   - Sentiment-specific vocabulary

2. **Feature Engineering & Preprocessing:**  
   - **TF-IDF vectorization** for traditional ML models  
   - **Tokenization** for transformer models (DistilBERT)

3. **Model Training & Evaluation:**  
   - Comparison of models based on **accuracy** and **training time**  

4. **Conclusions & Recommendations:**  
   - Insights on model performance  
   - Portfolio-ready demonstration of transformer usage

---

## 📂 File Structure

- `results/` → Model outputs and evaluation results (large, not included)
- `amazon_reviews_clean.csv` → Cleaned dataset (large, not included)
- `amazon_reviews_subset.csv` → Subset for training/testing (large, not included)
- `notebook.ipybn` → Jupyter notebooks for EDA, modeling, evaluation
- `README.md` → Project documentation

> **Note:** The `results/`, `amazon_reviews_clean.csv`, and `amazon_reviews_subset.csv` files are excluded from Git via `.gitignore` due to their size.  
> To reproduce these files, follow the preprocessing steps in the notebooks or scripts.

---

## 🗄️ Data

- Amazon product reviews spanning multiple years  
- Each review contains:  
  - Text of the review  
  - Rating  
  - Metadata (product & user IDs)  

- A smaller subset is used for **faster training** and portfolio purposes.

---

## ⚙️ How to Reproduce

1. Download the raw Amazon reviews dataset.  
2. Run the **preprocessing notebook/script** to create `amazon_reviews_clean.csv`.  
3. Optionally, generate `amazon_reviews_subset.csv` for quick experimentation.  
4. Train models using the **modeling notebook/script** to obtain results.

---

## 📊 Model Performance

| Model                  | Train Time        | Validation Accuracy | Test Accuracy | Comments |
|------------------------|-----------------|------------------|---------------|----------|
| Logistic Regression    | ~2.6 s            | 0.8673           | 0.8654        | Baseline TF-IDF model. Fast and stable. |
| SVM                    | ~1.8 s            | 0.8631           | 0.8606        | Lightweight and efficient. |
| DistilBERT             | 52 m 31.2 s     | 0.9334           | 0.9340        | Fine-tuned transformer. Significant improvement. |

**Key Conclusions:**

- **Baseline models (LR & SVM):** Quick, suitable for smaller datasets, ~86% accuracy.  
- **DistilBERT:** Outperforms classical models, ~93% accuracy, robust for NLP tasks.  
- **BERT-base:** Potentially stronger but more resource-heavy; recommended for future or cloud experiments.  

---

## 🛠️ Dependencies

- Python >= 3.13  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn  
- transformers (Hugging Face)  
- datasets (Hugging Face)  
- torch (PyTorch)  

---

## 📌 License

This project is for **educational and portfolio purposes**.
