import mlflow
import mlflow.sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
# import mlflow

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Sentiment_Analysis_Flipkart")

nltk.download("punkt")

data = pd.read_csv('data/clean_data')

X = data['clean_review']     # text data
y = data['sentiment']   


X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)



    # set experiment 1
mlflow.set_experiment("Sentiment_Analysis_Flipkart") 

with mlflow.start_run(run_name="NB_baseline"):

    # log parameters
    mlflow.log_param("model", "MultinomialNB")
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("alpha", 1.0)
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])


    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    mlflow.log_artifact("tfidf_vectorizer.pkl")

    # train model (your existing code)
    nb_model = MultinomialNB(alpha=1.0)
    nb_model.fit(X_train_tfidf, y_train)

    # predictions
    y_pred = nb_model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")
    # metrics
    mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
    mlflow.log_metric("f1_score", f1_score(y_test, y_pred, average="weighted"))
    mlflow.log_metric("precision", precision_score(y_test, y_pred, average="weighted"))
    mlflow.log_metric("recall", recall_score(y_test, y_pred, average="weighted"))


    # log model
    mlflow.sklearn.log_model(nb_model, "sentiment_model")


lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    return " ".join([lemmatizer.lemmatize(word) for word in text.split()])

X_train_lemma = X_train.apply(lemmatize_text)
X_test_lemma = X_test.apply(lemmatize_text)

tfidf_lemma = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf_lemma = tfidf_lemma.fit_transform(X_train_lemma)
X_test_tfidf_lemma = tfidf_lemma.transform(X_test_lemma)


mlflow.set_experiment("Sentiment_Analysis_Flipkart")

with mlflow.start_run(run_name="NB_lemmatized"):

    mlflow.log_param("model", "MultinomialNB")
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("preprocessing", "lemmatization")
    mlflow.log_param("lemmatizer", "WordNetLemmatizer")
    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("alpha", 1.0)


    joblib.dump(tfidf_lemma, "tfidf_lemmatized.pkl")
    mlflow.log_artifact("tfidf_lemmatized.pkl")

    # train model

    nb_model_lemma = MultinomialNB(alpha=1.0)
    nb_model_lemma.fit(X_train_tfidf_lemma, y_train)

    # predictions
    y_pred_lemma = nb_model_lemma.predict(X_test_tfidf_lemma)


    # compute & log metrics 
    accuracy = accuracy_score(y_test, y_pred_lemma)
    f1 = f1_score(y_test, y_pred_lemma, average="weighted")
    precision = precision_score(y_test, y_pred_lemma, average="weighted")
    recall = recall_score(y_test, y_pred_lemma, average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # log the model
    mlflow.sklearn.log_model(nb_model_lemma, "sentiment_model")





stemmer = PorterStemmer()

def stem_text(text):
    return " ".join([stemmer.stem(word) for word in text.split()])


X_train_stem = X_train.apply(stem_text)
X_test_stem = X_test.apply(stem_text)

tfidf_stem = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X_train_tfidf_stem = tfidf_stem.fit_transform(X_train_stem)
X_test_tfidf_stem = tfidf_stem.transform(X_test_stem)


mlflow.set_experiment("Sentiment_Analysis_Flipkart")

with mlflow.start_run(run_name="NB_stemmed"):
    mlflow.log_param("model", "MultinomialNB")
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("preprocessing", "stemming")
    mlflow.log_param("stemmer", "PorterStemmer")
    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("alpha", 1.0)

    joblib.dump(tfidf_stem, "tfidf_stemmed.pkl")
    mlflow.log_artifact("tfidf_stemmed.pkl")

    nb_model_stem = MultinomialNB(alpha=1.0)
    nb_model_stem.fit(X_train_tfidf_stem, y_train)

    y_pred_stem = nb_model_stem.predict(X_test_tfidf_stem)

    accuracy = accuracy_score(y_test, y_pred_stem)
    f1 = f1_score(y_test, y_pred_stem, average="weighted")
    precision = precision_score(y_test, y_pred_stem, average="weighted")
    recall = recall_score(y_test, y_pred_stem, average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    mlflow.sklearn.log_model(nb_model_stem, "sentiment_model")



mlflow.set_experiment("Sentiment_Analysis_Flipkart")

with mlflow.start_run(run_name="LR_tfidf_lemmatized"):

    # ---- Parameters ----
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("preprocessing", "lemmatization")
    mlflow.log_param("vectorizer", "TF-IDF")
    mlflow.log_param("ngram_range", "(1,2)")
    mlflow.log_param("max_features", 5000)

    mlflow.log_param("C", 1.0)
    mlflow.log_param("penalty", "l2")
    mlflow.log_param("solver", "liblinear")
    mlflow.log_param("max_iter", 1000)

    # --------------------
    # Train model
    # --------------------
    lr_model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=1000
    )

    lr_model.fit(X_train_tfidf, y_train)

    # --------------------
    # Evaluation
    # --------------------
    y_pred = lr_model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # --------------------
    # Artifacts
    # --------------------
    joblib.dump(tfidf, "tfidf_vectorizer.pkl")
    mlflow.log_artifact("tfidf_vectorizer.pkl")

    mlflow.sklearn.log_model(lr_model, "sentiment_model")


stemmer = PorterStemmer()

def stem_text(text):
    tokens = word_tokenize(text)
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(stemmed_tokens)

data["stemmed_review"] = data["clean_review"].apply(stem_text)

X = data["stemmed_review"]
y = data["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

mlflow.set_experiment("Sentiment_Analysis_Flipkart")

with mlflow.start_run(run_name="LR_tfidf_stemmed_v3"):


    # ---- Parameters ----
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
    mlflow.log_param("train_size", X_train.shape[0])
    mlflow.log_param("test_size", X_test.shape[0])

    # --------------------
    # Train model
    # --------------------
    lr_model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=1000
    )

    lr_model.fit(X_train_tfidf, y_train)


    # --------------------
    # Evaluation
    # --------------------
    y_pred = lr_model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    precision = precision_score(y_test, y_pred, average="weighted")
    recall = recall_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)

    # --------------------
    # Artifacts
    # --------------------
    mlflow.sklearn.log_model(lr_model, artifact_path="sentiment_model")
    joblib.dump(tfidf, "tfidf_vectorizer_stemmed.pkl")
    mlflow.log_artifact("tfidf_vectorizer_stemmed.pkl")



with mlflow.start_run(run_name="LR_DEBUG"):

    print("🚀 Run started")

    lr_model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=1000
    )

    lr_model.fit(X_train_tfidf, y_train)
    print("✅ Model trained")

    mlflow.sklearn.log_model(lr_model, artifact_path="sentiment_model")
    print("✅ Model logged")

    joblib.dump(tfidf, "tfidf_vectorizer_stemmed.pkl")
    mlflow.log_artifact("tfidf_vectorizer_stemmed.pkl")
    print("✅ Vectorizer logged")
