# Prefect + MLflow end-to-end pipeline for Sentiment Analysis
# -----------------------------------------------------------
# This flow orchestrates data loading, preprocessing (stemming),
# TF-IDF vectorization, Logistic Regression training, evaluation,
# and MLflow logging.

from prefect import flow, task
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

# --------------------
# Tasks
# --------------------

@task(retries=2, retry_delay_seconds=5)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


@task
def stem_text_series(text_series: pd.Series) -> pd.Series:
    stemmer = PorterStemmer()

    def _stem(text: str) -> str:
        tokens = word_tokenize(text)
        return " ".join(stemmer.stem(tok) for tok in tokens)

    return text_series.apply(_stem)


@task
def split_data(X: pd.Series, y: pd.Series):
    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


@task
def vectorize_text(X_train: pd.Series, X_test: pd.Series):
    tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    return X_train_tfidf, X_test_tfidf, tfidf


@task
def train_model(X_train_tfidf, y_train):
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=1000
    )
    model.fit(X_train_tfidf, y_train)
    return model


@task
def evaluate_and_log(
    model,
    tfidf,
    X_test_tfidf,
    y_test,
    run_name: str
):
    # MLflow setup
    mlflow.set_experiment("Sentiment_Analysis_Flipkart")

    with mlflow.start_run(run_name=run_name):
        # Parameters
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("preprocessing", "stemming")
        mlflow.log_param("stemmer", "PorterStemmer")
        mlflow.log_param("vectorizer", "TF-IDF")
        mlflow.log_param("ngram_range", "(1,2)")
        mlflow.log_param("max_features", 5000)
        mlflow.log_param("C", 1.0)
        mlflow.log_param("penalty", "l2")
        mlflow.log_param("solver", "liblinear")
        mlflow.log_param("max_iter", 1000)

        # Predictions
        y_pred = model.predict(X_test_tfidf)

        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)

        # Artifacts
        joblib.dump(tfidf, "tfidf_vectorizer_stemmed.pkl")
        mlflow.log_artifact("tfidf_vectorizer_stemmed.pkl")

        # Model
        mlflow.sklearn.log_model(model, artifact_path="sentiment_model")

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


# --------------------
# Flow
# --------------------

@flow(name="Flipkart_Sentiment_Prefect_Flow")
def sentiment_training_flow(data_path: str = "data/clean_data.csv"):
    df = load_data(data_path)

    X = stem_text_series(df["clean_review"])
    y = df["sentiment"]

    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train_tfidf, X_test_tfidf, tfidf = vectorize_text(X_train, X_test)

    model = train_model(X_train_tfidf, y_train)

    metrics = evaluate_and_log(
        model,
        tfidf,
        X_test_tfidf,
        y_test,
        run_name="LR_TFIDF_Stemmed_Prefect"
    )

    return metrics


if __name__ == "__main__":
    sentiment_training_flow()
