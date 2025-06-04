import os
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


def load_processed_dataset():
    path = os.path.normpath(os.path.expanduser("../data/processed_dataset/processed_dataset.csv"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found at {path}")
    return pd.read_csv(path)


def vectorize_texts(X_train, X_test, output_dir):
    excluded_stopwords = {'not', 'no', 'cannot', 'might','would', 'could', 'should', 'couldnt'}
    custom_stopwords = [word for word in ENGLISH_STOP_WORDS if word not in excluded_stopwords]
    vectorizer = TfidfVectorizer(
        stop_words=custom_stopwords,
        max_features=30000,
        ngram_range=(1, 3),
        min_df=2
    )

    X_train_vector = vectorizer.fit_transform(X_train)
    X_test_vector = vectorizer.transform(X_test)

    # Save vocabulary to JSON
    vocab_path = os.path.join(output_dir, "tfidf_vocabulary.json")
    vocab_cleaned = {}
    for k, v in vectorizer.vocabulary_.items():
        vocab_cleaned[k] = int(v)
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_cleaned, f, indent=2)

    print(f"\n{'\033[92m'}Vocabulary saved to:'\033[0m' {os.path.normpath(vocab_path)}")
    return X_train_vector, X_test_vector, vectorizer


def prepare_training():
    os.makedirs("../data/vectorizer", exist_ok=True)
    dataset = load_processed_dataset()
    X = dataset['review']
    y = dataset['sentiment']

    print("\nPreparing training & testing data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=91)

    X_train_vector, X_test_vector, vectorizer = vectorize_texts(X_train, X_test, "../data/vectorizer")

    return X_train_vector, X_test_vector, y_train, y_test, vectorizer