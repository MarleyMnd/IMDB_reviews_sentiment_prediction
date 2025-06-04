# IMDB Movie Reviews Sentiment Analysis

---

## Table of Contents

- [Project Structure](#project-structure)  
- [Requirements](#requirements)
- [Setup Instructions](#setup-instructions)
- [Important Notes](#important-notes)
- [Directory Structure Overview](#directory-structure-overview)

---


This project performs sentiment analysis on the IMDB movie reviews dataset using machine learning techniques. It downloads the dataset, preprocesses the reviews, visualizes the data, tunes hyperparameters, trains several classifiers, evaluates them, and allows user input for live sentiment prediction.

---

## Project Structure

- `main.py` — Main script that runs the entire pipeline: download, cleaning, visualization, training, and prediction.
- `A_download_dataset.py` — Downloads the IMDB dataset from Kaggle.
- `B_dataset_statistics.py` — Loads and prints dataset statistics.
- `C_data_cleaning.py` — Cleans and preprocesses the raw reviews (contraction expansion, lemmatization, duplicate removal).
- `D_data_visualization.py` — Visualizes dataset statistics (sentiment distribution, word count, word clouds).
- `E_prepare_training_testing_data.py` — Prepares train/test split and vectorizes the text data using TF-IDF.
- `F_hyperparameter_tuning.py` — Performs hyperparameter tuning with GridSearchCV for multiple models.
- `G_models_training_and_evaluation.py` — Trains and evaluates models, including cross-validation and metrics.
  
---

## Requirements

### Python Version
- Python 3.8 or higher recommended.

### Python Packages
A virtual environment is recommended, but optional.\
Install the following packages via `pip`:

```bash
pip install -r requirements.txt
```

### NLTK Resources
The script will download the following NLTK resources if not already present:
- wordnet
- own1-4
- averaged_perceptron_tagger
- punkt

---

## Setup Instructions
1. **Clone the repository** or copy files into your working directory.
2. Download the required **[python packages](#python-packages)**.
3. **Run** the main pipeline:
```bash
python main.py
```
This will:

- Download and verify the dataset.
- Print dataset statistics.
- Clean and preprocess the dataset (lemmatization, removing duplicates).
- Visualize data (sentiment distribution, word counts, word clouds).
- Prepare training and testing data (train/test split and TF-IDF vectorization).
- Perform hyperparameter tuning (SVM, Naive Bayes, Logistic Regression by default).
- Train and evaluate models.
- Enter interactive mode to input reviews and get sentiment predictions.

4. Interact with the model:\
Type your movie review (minimum 7 words) and press Enter.\
Type a digit from 0 to 9 to exit.

---

## Important Notes

* The Random Forest model creation is commented out by default in ``F_hyperparameter_tuning.py`` because it can be resource-intensive. You may enable it if your system has sufficient hardware resources by uncommenting line 35.

* The project excludes certain negation stopwords (not, no, etc.) during vectorization to better capture negation in reviews.

* If you want to **reprocess the data cleaning step**, delete the _**file**_ located at: ``../data/processed_dataset/processed_dataset.csv`` then rerun the main script.

---

## Directory Structure Overview

```
IMDB_reviews/
└── data/
    └── hyperparameters/
        └── models' hyperparameters stored here after first run
    ├── processed_dataset/
        └── processed data stored here after first run
    ├── raw_dataset/
        └── raw data stored here after first run
    ├── vectorizer/
        └── vectorizer's vocabulary stored here after first run
└── src/
    ├── main.py
    ├── A_download_dataset.py
    ├── B_dataset_statistics.py
    ├── C_data_cleaning.py
    ├── D_data_visualization.py
    ├── E_prepare_training_testing_data.py
    ├── F_hyperparameter_tuning.py
    ├── G_models_training_and_evaluation.py
```