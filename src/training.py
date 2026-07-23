import os
import time
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from src.preprocessing import preprocess_pipeline

DATASET_PATH = os.path.join('dataset', 'mahasiswa.csv')
MODEL_PATH = os.path.join('model', 'naive_bayes.pkl')
VECTORIZER_PATH = os.path.join('model', 'vectorizer.pkl')


def load_dataset(path=None):
    if path is None:
        path = DATASET_PATH
    df = pd.read_csv(path)
    df = df.dropna(subset=['text', 'sentiment'])
    df['text'] = df['text'].astype(str)
    df['sentiment'] = df['sentiment'].str.strip().str.lower()
    return df


def train_model(df=None):
    if df is None:
        df = load_dataset()

    os.makedirs('model', exist_ok=True)

    df['processed'] = df['text'].apply(preprocess_pipeline)

    X = df['processed']
    y = df['sentiment']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)

    start = time.time()
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    elapsed = round(time.time() - start, 4)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)

    return model, vectorizer, X_train, X_test, y_train, y_test, elapsed


def load_model():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(VECTORIZER_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def get_training_info():
    df = load_dataset()
    train_size = int(len(df) * 0.8)
    test_size = len(df) - train_size
    model, vectorizer = load_model()
    vocab_size = len(vectorizer.vocabulary_) if vectorizer else 0
    return {
        'total': len(df),
        'train_size': train_size,
        'test_size': test_size,
        'vocab_size': vocab_size,
    }
