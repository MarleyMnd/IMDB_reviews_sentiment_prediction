from A_download_dataset import download_imdb_dataset
from B_dataset_statistics import statistics
from C_data_cleaning import clean_data
from D_data_visualization import visualize_data
from E_prepare_training_testing_data import prepare_training
from F_hyperparameter_tuning import tune_hyperparameters
from G_models_training_and_evaluation import train_and_evaluate_models

def predict_user_review(model, vectorizer):
    while True:
        review = input("Enter a review (enter digit 0-9 to exit):\n>>> ")
        if review.isdigit() and len(review) == 1:
            print("Exiting.")
            break

        word_count = len(review.strip().split())
        if word_count < 7:
            print(f"Review {'\033[93m'}TOO SHORT{'\033[0m'} (less than 7 words). Please write a longer review.")
            continue

        review_vector = vectorizer.transform([review])
        prediction = model.predict(review_vector)[0]

        if prediction == 1:
            print(f"Review was classified as {'\033[92m' + '\033[1m' + '\033[4m'}POSITIVE{'\033[0m'}.")
        else:
            print(f"Review was classified as {'\033[91m' + '\033[1m' + '\033[4m'}NEGATIVE{'\033[0m'}.")


def main():
    print(f"\n{'\033[1m' + '\033[4m' + '\033[94m'}Download dataset{'\033[0m'}")
    download_imdb_dataset()

    print(f"\n\n{'\033[1m' + '\033[4m' + '\033[94m'}Dataset statistics{'\033[0m'}")
    statistics()

    print(f"\n\n{'\033[1m' + '\033[4m' + '\033[94m'}Data cleaning{'\033[0m'}")
    clean_data()

    print(f"\n\n{'\033[1m' + '\033[4m' + '\033[94m'}Data visualisation{'\033[0m'}")
    visualize_data()

    print(f"\n\n{'\033[1m' + '\033[4m' + '\033[94m'}Train/test split & vectorization{'\033[0m'}")
    X_train_vector, X_test_vector, y_train, y_test, vectorizer = prepare_training()

    print(f"\n\n{'\033[1m' + '\033[4m' + '\033[94m'}Hyperparameter tuning{'\033[0m'}")
    best_models = tune_hyperparameters(X_train_vector, y_train)

    print(f"\n\n{'\033[1m' + '\033[4m' + '\033[94m'}Models training{'\033[0m'}")
    trained_models = train_and_evaluate_models(best_models, X_train_vector, y_train, X_test_vector, y_test)

    print(f"\n\n{'\033[1m' + '\033[4m' + '\033[94m'}User review prediction{'\033[0m'}")
    best_model = trained_models["Logistic Regression"] # SVM || Naive Bayes || Logistic Regression || Random Forest (only available if activated in F_hyperparameter_tuning.py line 35)
    predict_user_review(best_model, vectorizer)

if __name__ == "__main__":
    main()
