import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS


def load_processed_dataset():
    path = os.path.normpath(os.path.expanduser("../data/processed_dataset/processed_dataset.csv"))
    if not os.path.exists(path):
        raise FileNotFoundError(f"Processed dataset not found at {path}")
    return pd.read_csv(path)


def plot_sentiment_distribution(dataset):
    print("\nPlotting sentiment distribution...")
    sentiment_map = {1: "Positive", 0: "Negative"}
    sentiment_counts = dataset['sentiment'].value_counts().rename(index=sentiment_map)

    sentiment_counts.plot(kind='bar', color=['green', 'red'])
    plt.title("Reviews distribution")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of reviews")
    plt.tight_layout()
    plt.show()
    print(f"{'\033[92m'}Completed.{'\033[0m'}")


def plot_word_count_distribution(dataset):
    print("\nPlotting word count distribution...")
    dataset['word_count'] = dataset['review'].apply(lambda x: len(x.split()))

    sns.boxplot(x=dataset['sentiment'], y=dataset['word_count'])
    plt.title("Number of words per type of review")
    plt.xlabel("Sentiment (0 = negative, 1 = positive)")
    plt.ylabel("Number of words")
    plt.tight_layout()
    plt.show()
    print(f"{'\033[92m'}Completed.{'\033[0m'}")


def generate_wordclouds(dataset):
    positive_review = ' '.join(dataset[dataset['sentiment'] == 1]['review'])
    negative_review = ' '.join(dataset[dataset['sentiment'] == 0]['review'])

    stopwords_pos = set(STOPWORDS)
    stopwords_pos.update(['movie', 'film', 'character', 'make', 'one', 'even', 'time', 'see', 'much', 'made', 'go', 'show', 's', 'watch', 'think', 'will'])
    stopwords_neg = set(STOPWORDS)
    stopwords_neg.update(['movie', 'film', 'character', 'make', 'one', 'even', 'time', 'see', 'much', 'made', 'well', 'go', 'show', 'great', 'good', 's', 'will', 'really', 'give', 'watch', 'think', 'look', 'say', 'know'])

    print("\nGenerating positive reviews wordcloud...")
    wc_pos = WordCloud(width=400, height=300, background_color='white', stopwords=stopwords_pos).generate(positive_review)

    plt.imshow(wc_pos, interpolation='bilinear')
    plt.axis('off')
    plt.title("Most frequent words in positive reviews")
    plt.tight_layout()
    plt.show()
    print(f"{'\033[92m'}Completed.{'\033[0m'}")

    print("\nGenerating negative reviews wordcloud...")
    wc_neg = WordCloud(width=400, height=300, background_color='black', colormap='Reds', stopwords=stopwords_neg).generate(negative_review)

    plt.imshow(wc_neg, interpolation='bilinear')
    plt.axis('off')
    plt.title("Most frequent words in negative reviews")
    plt.tight_layout()
    plt.show()
    print(f"{'\033[92m'}Completed.{'\033[0m'}")


def visualize_data():
    dataset = load_processed_dataset()
    plot_sentiment_distribution(dataset)
    plot_word_count_distribution(dataset)
    generate_wordclouds(dataset)
