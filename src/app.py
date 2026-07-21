import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

# Load model and vectorizer
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


def clean_text(text):

    text = str(text).lower()

    text = re.sub(r"http\S+|www\S+", "", text)

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\d+", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

st.title(" Fake News Detection")

st.write("Enter a news article below to check whether it is Real or Fake.")

news = st.text_area("News Article")



if st.button("Predict"):

    cleaned_news = clean_text(news)

    vector = vectorizer.transform([cleaned_news])

    prediction = model.predict(vector)

    probability = model.predict_proba(vector)

    if prediction[0] == 0:
        st.error("Fake News ")
        st.write(f"Confidence: {probability[0][0]*100:.2f}%")
    else:
        st.success("Real News ")
        st.write(f"Confidence: {probability[0][1]*100:.2f}%")