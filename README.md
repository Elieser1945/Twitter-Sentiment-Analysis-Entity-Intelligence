# 📊 Twitter Sentiment Analysis & Entity Intelligence

<div align="center">

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://twitter-sentiment-analysis-entity-intelligence-by-elieser.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Library-Scikit_Learn-orange.svg)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLP-NLTK-green.svg)](https://www.nltk.org/)

An end-to-end Machine Learning and Natural Language Processing (NLP) web application designed to analyze public social media sentiments associated with specific entities, delivering actionable business intelligence and automated operational recommendations.

[**Explore Live Demo 🚀**](https://twitter-sentiment-analysis-entity-intelligence-by-elieser.streamlit.app/)

</div>

---

## 👨‍💻 Developed by
* **Elieser Pasaribu**
* **Role:** Data Analyst | Data Scientist | Machine Learning Engineer

---

## 🎯 Project Overview
In modern business environments, tracking public opinion and customer sentiment at scale is crucial for brand protection, product development, and customer satisfaction. This project automates the extraction and classification of public opinions from text data. 

Unlike standard sentiment models that only output classifications, this system integrates an **Actionable Business Intelligence layer** that automatically translates negative feedback into operational recommendations for Product, R&D, and Customer Support teams.

### Key Capabilities:
1. **Real-Time Sentiment Prediction:** Instant sentiment evaluation (Positive, Negative, Neutral, Irrelevant) for single text inputs combined with topic entities.
2. **Bulk Batch Processing:** Capability to upload CSV datasets containing thousands of records for automated mass sentiment scoring and executive summaries.
3. **Automated Business Insights:** Generates strategic operational recommendations based on detected customer friction points.

---

## 🛠️ System Architecture & NLP Pipeline
The machine learning pipeline is structured into a robust, end-to-end workflow:
* **1. Raw Text & Entity Input:** Captures user queries combining target entities and tweet statements.
* **2. Advanced Preprocessing:** Executes tokenization, punctuation removal, stopword filtering, POS tagging, and WordNet Lemmatization.
* **3. Feature Extraction:** Transforms textual tokens into numerical representations using custom `CountVectorizer` and `TfidfTransformer`.
* **4. Supervised Machine Learning:** Leverages optimized classification models (`RandomForestClassifier`) to predict sentiment classes.
* **5. Intelligence & Action Layer:** Delivers real-time classification results paired with automated business recommendations.

---

## 📂 Dataset Information
* **Source:** Kaggle / Global Twitter Entity Sentiment Dataset.
* **Format:** Multi-class text dataset consisting of entity categories, sentiment labels, and tweet contents.
* **Preprocessing Steps:** Handling missing values, duplicate removal, text cleaning, and outlier filtering using IQR.

---

## 🚀 Model Evaluation & Performance
Several baseline and ensemble algorithms were evaluated on the dataset:
* **KNeighborsClassifier:** High training performance.
* **RandomForestClassifier:** Selected for production due to its robust generalization, stability, and fast inference time.
* **Logistic Regression & Decision Tree:** Used as comparative baselines.
* **Evaluation Metrics:** Evaluated using Accuracy, Precision, Recall, and F1-Score across all sentiment classes.

---

## 💻 Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Pipeline, TF-IDF, RandomForest)
* **Natural Language Processing:** NLTK (WordNetLemmatizer, POS Tagger, Stopwords)
* **Web Application Framework:** Streamlit
* **Model Serialization:** Joblib

---

## ⚙️ Local Installation & Setup
If you want to run this project locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/elieser-pasaribu/twitter-sentiment-analysis.git
   cd twitter-sentiment-analysis
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate   # For Windows
   # source .venv/bin/activate  # For macOS/Linux
   ```

3. **Install dependencies:**
   ```bash
   pip install streamlit pandas joblib scikit-learn nltk
   ```

4. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

---

## 📷 Application Preview
* **Single Text Prediction:** Users can input an entity (e.g., Samsung) and a review statement to get immediate sentiment insights and business recommendations.
* **Batch Processing:** Upload a CSV file with entity and tweet content columns to evaluate mass datasets instantly.

---

## 📫 Connect with Me
If you have any questions, feedback, or collaboration opportunities, feel free to reach out:

* **Live App:** [Streamlit Demo](https://twitter-sentiment-analysis-entity-intelligence-by-elieser.streamlit.app/)
* **Developer:** Elieser Pasaribu
