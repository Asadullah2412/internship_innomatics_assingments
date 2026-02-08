# Flipkart Sentiment Analysis – End-to-End MLOps Pipeline

This project demonstrates a **production-style MLOps workflow** for sentiment analysis using real-world tools such as **scikit-learn, MLflow, and Prefect**. The focus is not just on model accuracy, but on **experiment tracking, model registry, orchestration, and reproducibility**.

---

## 📌 Project Overview

The goal of this project is to build a sentiment classification model for Flipkart product reviews and manage the full ML lifecycle:

* Text preprocessing and feature engineering
* Model training and evaluation
* Experiment tracking with MLflow
* Model registration, tagging, and stage promotion
* Workflow orchestration with Prefect
* Monitoring runs via MLflow UI and Prefect Dashboard

---

## 🧠 Model Details

* **Algorithm**: Logistic Regression
* **Vectorization**: TF-IDF (unigrams + bigrams)
* **Text Processing**:

  * Cleaning
  * Tokenization
  * Porter Stemming
* **Evaluation Metrics**:

  * Accuracy
  * Precision
  * Recall
  * F1-score (used for best model selection)

---

## 🛠️ Tech Stack

* Python
* scikit-learn
* pandas, numpy
* NLTK
* MLflow
* Prefect 2.x
* Matplotlib

---

## 📂 Project Structure

```
mlflow_sentiment/
│
├── data/
│   └── clean_data.csv
│
├── notebooks/
│   └── experimentation.ipynb
│
├── workflow.py              # Prefect workflow
├── requirements.txt
├── README.md
└── mlruns/                  # MLflow tracking directory
```

---

## 🔬 Experiment Tracking (MLflow)

MLflow is used to track:

* Model hyperparameters
* Preprocessing choices
* Evaluation metrics
* Artifacts (TF-IDF vectorizer, trained model)

### Logged Metrics

* Accuracy
* Precision (weighted)
* Recall (weighted)
* F1-score (weighted)

Metric plots are visualized directly in the **MLflow UI**.

---

## 🏆 Model Registry & Versioning

The best-performing model (based on **F1-score**) is:

* Registered in **MLflow Model Registry**
* Tagged with metadata (algorithm, preprocessing, dataset version)
* Promoted to stages such as:

  * `Staging`
  * `Production`

This enables controlled model lifecycle management.

---

## 🔄 Workflow Orchestration (Prefect)

A Prefect flow is used to orchestrate the training pipeline:

### Key Tasks

* Load dataset
* Preprocess text
* Vectorize data
* Train model
* Evaluate metrics
* Log results to MLflow

### Why Prefect?

* Task-level retries
* Clear logging
* Visual DAG
* Easy scheduling
* Production-grade orchestration

---

## ⏰ Scheduling

The Prefect workflow can be scheduled (e.g., daily retraining) using:

* Prefect deployments
* Cron or interval schedules

The flow execution and task states can be monitored via the **Prefect Dashboard**.

---

## 📊 Dashboards

* **MLflow UI**: Experiment comparison, metrics, artifacts, model registry
* **Prefect UI**: Flow runs, retries, logs, and task-level status

Screenshots of both dashboards are included in the project documentation.

---

## ▶️ How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 2️⃣ Start MLflow UI

```bash
mlflow ui
```

### 3️⃣ Start Prefect Server

```bash
prefect server start
```

### 4️⃣ Run the workflow

```bash
python workflow.py
```

---

## 🚀 Key Learnings

* Proper experiment tracking is critical for reproducibility
* Model registry enables safe promotion to production
* Relative paths break in orchestration — absolute paths matter
* Prefect provides clear visibility and reliability for ML pipelines

---

## 📌 Future Improvements

* Add hyperparameter tuning with multiple runs
* Add data validation checks
* Integrate model inference pipeline
* Deploy model as an API

---

## 👤 Author

Built as part of a hands-on **MLOps learning project** focused on real-world tooling and best practices.
