import streamlit as st
import re
import nltk
from nltk.corpus import stopwords
import joblib

st.set_page_config(page_title="Sentiment Analysis", layout="centered")


st.title("Sentiment Analysis App")
st.write("Enter a review and the model will predict its sentiment.")


# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text



@st.cache_resource
def load_model():
    model = joblib.load("sentiment_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer


model, tfidf = load_model()


st.success("Model loaded successfully ✅")

st.subheader("🔍 Try a review")

user_input = st.text_area("Enter your review text here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_text = clean_text(user_input)
        vectorized_text = tfidf.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]


    if prediction == "Positive":
        st.success("😊 Sentiment: Positive")
    elif prediction == "Neutral":
        st.info("😐 Sentiment: Neutral")
    else:
        st.error("😠 Sentiment: Negative")

st.markdown("---")
st.caption("Streamlit app using a saved Naive Bayes model")