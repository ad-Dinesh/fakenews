# 📰 Fake News Detection using Machine Learning

A Machine Learning and Natural Language Processing (NLP) project that classifies news articles as **Real** or **Fake** using **TF-IDF Vectorization** and **Logistic Regression**.

The application provides a simple Streamlit web interface where users can enter a news article and receive a prediction along with the model's confidence score.

---

## 🚀 Features

- Detects Fake and Real news articles
- NLP-based text preprocessing
- TF-IDF feature extraction
- Logistic Regression classifier
- Confidence score prediction
- Streamlit web application
- Clean and modular Python project structure

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- NLTK
- Joblib
- Streamlit

---

## 📂 Project Structure

```
Fake-News-Detection/
│
├── dataset/
│   ├── Fake.csv
│   ├── True.csv
│   └── processed_news.csv
│
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📊 Dataset

This project uses the **Fake and Real News Dataset** containing thousands of news articles labeled as Fake or Real.

Dataset Columns

- title
- text
- subject
- date
- label

---

## ⚙️ Machine Learning Pipeline

1. Load Dataset
2. Merge Fake & Real datasets
3. Data Cleaning
4. Text Preprocessing
5. TF-IDF Vectorization
6. Train-Test Split
7. Logistic Regression Training
8. Model Evaluation
9. Save Model
10. Streamlit Deployment

---

## 🧹 Text Preprocessing

The following preprocessing steps are applied:

- Convert text to lowercase
- Remove URLs
- Remove punctuation
- Remove numbers
- Remove stopwords
- Remove extra whitespace

---

## 🤖 Model

Algorithm

- Logistic Regression

Feature Extraction

- TF-IDF Vectorizer

---

## 📈 Model Performance

| Metric | Value |
|---------|------:|
| Training Accuracy | **97.49%** |
| Testing Accuracy | **98.43%** |

---

## ▶️ Installation

Clone the repository

```bash
git clone https://github.com/ad-Dinesh/fake-news-detection.git
```

Move into the project

```bash
cd fake-news-detection
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run preprocessing

```bash
python src/preprocess.py
```

Train the model

```bash
python src/train.py
```

Run prediction

```bash
python src/predict.py
```

Launch Streamlit App

```bash
streamlit run app.py
```

---

## 💡 Example Prediction

Input

```
NASA successfully launched a new satellite to monitor climate change.
```

Output

```
Prediction:
Real News ✅

Confidence:
97.81%
```



## 🔮 Future Improvements

- BERT / RoBERTa based classifier
- Real-time fact verification
- News source credibility analysis
- Multi-language news detection
- Explainable AI predictions

---

## 👨‍💻 Author

**Dinesh Dharavath**

GitHub: https://github.com/ad-Dinesh

LinkedIn:
https://www.linkedin.com/in/dinesh-dharavath-b176a2342/

Email:
dineshdharavath03@gmail.com

---

## ⭐ If you found this project useful, consider giving it a star.
