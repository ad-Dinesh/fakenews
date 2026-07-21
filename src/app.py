import streamlit as st
import joblib
import re
import string
import nltk
from nltk.corpus import stopwords

# ---------------- Page Configuration ---------------- #
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ---------------- Download Stopwords ---------------- #
try:
    stop_words = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    stop_words = set(stopwords.words("english"))

# ---------------- Load Model ---------------- #
model = joblib.load("model/model.pkl")
vectorizer = joblib.load("model/vectorizer.pkl")


# ---------------- Text Cleaning Function ---------------- #
def clean_text(text):
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Remove numbers
    text = re.sub(r"\d+", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]

    return " ".join(words)


# ---------------- UI ---------------- #
st.title("📰 Fake News Detection using Machine Learning")

st.markdown("""
This application uses **TF-IDF Vectorization** and **Logistic Regression**
to classify news articles as **Real** or **Fake**.
""")

news = st.text_area(
    "Enter News Article",
    height=250,
    placeholder="Paste any news article here..."
)

# ---------------- Prediction ---------------- #
if st.button("Predict"):

    if news.strip() == "":
        st.warning("Please enter a news article.")
    else:

        cleaned_news = clean_text(news)

        vector = vectorizer.transform([cleaned_news])

        prediction = model.predict(vector)

        probability = model.predict_proba(vector)

        st.subheader("Prediction Result")

        if prediction[0] == 0:
            st.error("🚨 Fake News")
            st.write(f"**Confidence:** {probability[0][0] * 100:.2f}%")
        else:
            st.success("✅ Real News")
            st.write(f"**Confidence:** {probability[0][1] * 100:.2f}%")

# ---------------- Sidebar ---------------- #
st.sidebar.header("About Project")

st.sidebar.markdown("""
### Model
- Logistic Regression

### Feature Extraction
- TF-IDF Vectorizer

### Dataset
- Fake.csv
- True.csv

### Language
- Python

### Framework
- Streamlit
""")

st.markdown("---")
st.caption("Developed by Dinesh | Machine Learning Project")