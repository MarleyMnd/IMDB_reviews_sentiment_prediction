import os, re
import time
import pandas as pd
import nltk
import contractions
from tqdm import tqdm
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

def download_nltk_resources():
    print("\nDownloading NLTK resources:")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('averaged_perceptron_tagger_eng')
    nltk.download('punkt_tab')

def get_dataset_paths():
    raw_path = os.path.normpath(os.path.expanduser(
        "../data/raw_dataset/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/versions/1/IMDB Dataset.csv"))
    save_dir = os.path.normpath(os.path.expanduser("../data/processed_dataset"))
    save_path = os.path.join(save_dir, "processed_dataset.csv")
    os.makedirs(save_dir, exist_ok=True)
    return raw_path, save_path

def lemmatize_review(review_tokens):
    lemma = WordNetLemmatizer()
    tagged = pos_tag(review_tokens)
    lemmatized = []
    for word, tag in tagged:
        if tag.startswith('N'):
            wn_tag = wordnet.NOUN
        elif tag.startswith('J'):
            wn_tag = wordnet.ADJ
        elif tag.startswith('V'):
            wn_tag = wordnet.VERB
        elif tag.startswith('R'):
            wn_tag = wordnet.ADV
        else:
            wn_tag = wordnet.NOUN
        lemmatized.append(lemma.lemmatize(word, pos=wn_tag))
    return " ".join(lemmatized)

def clean_and_process_dataset(raw_dataset):
    reviews = raw_dataset['review'].str.replace(r"<br\s*/?>", " ", regex=True)
    expanded_reviews = []
    for review in tqdm(reviews, desc="Cleaning reviews"):
        review = review.lower()
        review = contractions.fix(review)
        expanded_reviews.append(review)
    reviews = pd.Series(expanded_reviews)
    time.sleep(2)
    print("Please wait... Lemmatisation is about to start...")
    reviews = reviews.str.replace(r"[^a-zA-Z]", " ", regex=True)
    reviews = reviews.apply(word_tokenize)

    lemmatized_reviews = []
    for review in tqdm(reviews, desc="Lemmatizing reviews"):
        lemmatized_reviews.append(lemmatize_review(review))

    processed_dataset = pd.DataFrame({ "review": lemmatized_reviews, "sentiment": raw_dataset['sentiment'].map({'positive': 1, 'negative': 0})})

    nb_reviews_before = len(processed_dataset)
    processed_dataset = processed_dataset.drop_duplicates(subset="review").reset_index(drop=True)
    nb_reviews_after = len(processed_dataset)
    print(f"\nRemoved {nb_reviews_before - nb_reviews_after} duplicate reviews during cleaning.")

    return processed_dataset

def clean_data():
    raw_path, save_path = get_dataset_paths()

    if not os.path.exists(save_path):
        print(f"\n{'\033[92m'}Processing dataset...{'\033[0m'}")
        raw_dataset = pd.read_csv(raw_path)
        download_nltk_resources()
        time.sleep(1)
        processed_dataset = clean_and_process_dataset(raw_dataset)
        processed_dataset.to_csv(save_path, index=False)
        print(f"\n{'\033[92m'}Processed dataset stored in:{'\033[0m'} {save_path}\n")
    else:
        print(f"\n{'\033[93m'}Dataset already processed and saved in:{'\033[0m'} {save_path}")
        print(f"{'\033[93m'}To re-process, delete the file and rerun.{'\033[0m'}\n")
        time.sleep(1)

    processed_dataset = pd.read_csv(save_path)
    raw_dataset = pd.read_csv(raw_path)

    print("Sample comparison:")
    print("Review 1:")
    print("Original:", raw_dataset['review'][1])
    print("Processed:", processed_dataset['review'][1])
    print("Review 2:")
    print("Original:", raw_dataset['review'][2])
    print("Processed:", processed_dataset['review'][2])
