import time
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report, confusion_matrix

def train_and_evaluate_models(best_models, X_train_vector, y_train, X_test_vector, y_test):
    trained_models = {}

    for name, model in best_models.items():
        print(f"\n{'\033[1m' + '\033[4m'}{name}{'\033[0m'} — Cross-validation on training data")
        start = time.time()
        cv_scores = cross_val_score(model, X_train_vector, y_train, cv=5, scoring='f1_macro', n_jobs=-1)
        elapsed = time.time() - start
        print(f"{'\033[92m'}Average F1-score (5-fold): {cv_scores.mean():.4f}{'\033[0m'}")
        print(f"Execution time: {elapsed:.2f} seconds")

        print(f"\n{name} - Training final model on full training set...")
        model.fit(X_train_vector, y_train)

        print(f"{name} - Predicting on test set...")
        y_pred = model.predict(X_test_vector)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(f"Confusion Matrix - {name}")
        plt.show()

        trained_models[name] = model

    return trained_models
