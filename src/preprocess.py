import pandas as pd

# Load datasets
fake_df = pd.read_csv("dataset/Fake.csv")
true_df = pd.read_csv("dataset/True.csv")

# Display basic information
print("=" * 50)
print("FAKE NEWS DATASET")
print("=" * 50)
print(fake_df.head())

print("\nDataset Shape:", fake_df.shape)
print("\nColumns:")
print(fake_df.columns)

print("\nMissing Values:")
print(fake_df.isnull().sum())

print("\n" + "=" * 50)
print("REAL NEWS DATASET")
print("=" * 50)
print(true_df.head())

print("\nDataset Shape:", true_df.shape)
print("\nColumns:")
print(true_df.columns)

print("\nMissing Values:")
print(true_df.isnull().sum())

# Dataset information
print("\nFake Dataset Info")
print(fake_df.info())

print("\nTrue Dataset Info")
print(true_df.info())

# Add labels
fake_df["label"] = 0
true_df["label"] = 1

print(fake_df.head())
print(true_df.head())


# Merge fake and real datasets
news_df = pd.concat([fake_df, true_df], axis=0)

# Reset index
news_df = news_df.reset_index(drop=True)

print("\nMerged Dataset")
print(news_df.head())

print("\nDataset Shape:", news_df.shape)

import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))

def clean_text(text):
    # Convert to lowercase
    text = text.lower()

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

news_df["text"] = news_df["text"].apply(clean_text)
print(news_df["text"].head())

# Features and Labels
X = news_df["text"]
y = news_df["label"]

print("Features Shape:", X.shape)
print("Labels Shape:", y.shape)

# Save processed dataset
news_df.to_csv("dataset/processed_news.csv", index=False)

print("\nProcessed dataset saved successfully!")