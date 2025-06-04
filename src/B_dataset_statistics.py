import pandas as pd
import os


def load_raw_reviews_dataset():
    dataset_path = os.path.normpath(os.path.expanduser(
        "../data/raw_dataset/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/versions/1/IMDB Dataset.csv"))
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found in: {dataset_path}")
    return pd.read_csv(dataset_path)


def print_dataset_statistics(dataset):
    print("\nDataset info:")
    print(dataset.info())

    print("\nDataset statistics:")
    print(dataset.describe(include='all'))


def statistics():
    dataset = load_raw_reviews_dataset()
    print_dataset_statistics(dataset)