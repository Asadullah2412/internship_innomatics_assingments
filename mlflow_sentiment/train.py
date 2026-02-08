import mlflow
import mlflow.sklearn
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
nltk.download('wordnet')
nltk.download('omw-1.4')


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



    # set experiment
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

