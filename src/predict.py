import os
import joblib

# Paths to the saved model and vectorizer
MODEL_PATH = os.path.join("model", "model.pkl")
VECTORIZER_PATH = os.path.join("model", "vectorizer.pkl")


def load_artifacts(model_path, vectorizer_path):
    """Load the trained model and vectorizer from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(vectorizer_path):
        raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


def predict_news(news, model, vectorizer):
    """Predict whether a news article is Fake or Real."""
    if not news or not news.strip():
        return "No input provided."

    # Convert text to vector using the same vectorizer used during training
    news_vector = vectorizer.transform([news])

    # Predict
    prediction = model.predict(news_vector)

    return "Fake News" if prediction[0] == 0 else "Real News"


def main():
    try:
        model, vectorizer = load_artifacts(MODEL_PATH, VECTORIZER_PATH)
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        return

    news = input("Enter News: ")
    result = predict_news(news, model, vectorizer)
    print("\nPrediction:", result)


if __name__ == "__main__":
    main()