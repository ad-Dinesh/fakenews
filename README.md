# Fake News Detection 📰🔍

Classify news articles as **Real** or **Fake** in seconds — powered by TF-IDF + Logistic Regression, wrapped in a clean Streamlit UI.

`Python` `Scikit-learn` `NLTK` `Streamlit` `TF-IDF` `Logistic Regression`

---

## Why this project?

Misinformation spreads faster than fact-checkers can keep up. This project takes a classic, lightweight NLP approach — no heavyweight transformers, no GPU required — and still lands **~98% test accuracy**. Paste an article, get a verdict, get a confidence score. That's it.

```
Input:  "NASA successfully launched a new satellite to monitor climate change."
Output: Real News ✅  |  Confidence: 97.81%
```

---

## How it works

```
raw text ──▶ clean & normalize ──▶ TF-IDF vectorize ──▶ Logistic Regression ──▶ label + confidence
```

1. **Ingest** — merge labeled Fake/Real news datasets
2. **Clean** — lowercase, strip URLs, punctuation, numbers, stopwords
3. **Vectorize** — TF-IDF turns text into weighted numeric features
4. **Train** — Logistic Regression learns the decision boundary
5. **Serve** — Streamlit app for instant predictions

---

## Results

| | Score |
|---|---|
| Training Accuracy | **98.34** |
| **Testing Accuracy** | **98.43%** |

---

## Get it running

```bash
git clone https://github.com/ad-Dinesh/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
```

Then walk the pipeline end to end:

```bash
python src/preprocess.py   # clean + merge the raw CSVs
python src/train.py        # train + save the model
python src/predict.py      # quick CLI sanity check
streamlit run app.py       # launch the web app
```

---

## What's inside

```
Fake-News-Detection/
├── dataset/            raw + processed news data
├── model/               trained model.pkl + vectorizer.pkl
├── src/
│   ├── preprocess.py    cleans & merges the dataset
│   ├── train.py         TF-IDF + Logistic Regression training
│   └── predict.py       loads model, scores new text
├── app.py                Streamlit interface
└── requirements.txt
```

---

## The stack

- **Pandas / NumPy** — data wrangling
- **NLTK** — stopword removal & text cleanup
- **Scikit-learn** — TF-IDF vectorizer + Logistic Regression
- **Joblib** — model persistence
- **Streamlit** — the front end

---

## Dataset

Built on the **Fake and Real News Dataset**, with columns: `title`, `text`, `subject`, `date`, `label`.

---

## Roadmap

- [ ] Swap in BERT / RoBERTa for deeper contextual understanding
- [ ] Real-time fact verification against trusted sources
- [ ] Source credibility scoring
- [ ] Multi-language support
- [ ] Explainable predictions (which words tipped the model?)

---

## Author

**Dinesh Dharavath**
[GitHub](https://github.com/ad-Dinesh) · [LinkedIn](https://www.linkedin.com/in/dinesh-dharavath-b176a2342/) · dineshdharavath03@gmail.com

---

⭐ **If this was useful, a star helps more than you'd think.**
