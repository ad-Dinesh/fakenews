import pandas as pd
# Load processed dataset
news_df = pd.read_csv("dataset/processed_news.csv")

print(news_df.head())
print(news_df.shape)

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load processed dataset
news_df = pd.read_csv("dataset/processed_news.csv")

# Features and Labels
X = news_df["text"]
y = news_df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data :", X_train.shape)
print("Testing Data :", X_test.shape)