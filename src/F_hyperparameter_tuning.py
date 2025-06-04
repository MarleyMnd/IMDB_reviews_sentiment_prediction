import os
import time
import json
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def tune_hyperparameters(X_train_vector, y_train):
    hyperparameters_grids = {
        "SVM": {
            'C': [0.01, 0.1, 1, 5, 10]
        },
        "Naive Bayes": {
            'alpha': [0.5, 1.0, 1.5]
        },
        "Logistic Regression": {
            'C': [0.01, 0.1, 1, 5, 10],
            'penalty': ['l2'],
            'solver': ['lbfgs']
        },
        "Random Forest": {
            'n_estimators': [100, 200, 300],
            'max_depth': [None, 20, 50],
            'max_features': ['sqrt', 'log2']
        }
    }

    models = {
        "SVM": LinearSVC(random_state=91, max_iter=5000),
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=91),
        # You should uncomment only if you have enough hardware resources:
        # "Random Forest": RandomForestClassifier(random_state=91, n_jobs=-1)
    }

    param_file = os.path.normpath("../data/hyperparameters/best_hyperparameters.json")
    best_models = {}
    saved_params = {}

    if os.path.exists(param_file) and os.stat(param_file).st_size > 0:
        print("\nLoading previously saved hyperparameters...")
        with open(param_file, 'r') as f:
            saved_params = json.load(f)

        models_to_be_removed = [name for name in saved_params if name not in models]
        if models_to_be_removed:
            print(", ".join(models_to_be_removed))
            for name in models_to_be_removed:
                del saved_params[name]

    missing_models = [name for name in models if name not in saved_params]

    if missing_models:
        print(f"{'\033[93m'}Some models are missing hyperparameters. Running GridSearchCV for:{'\033[0m'}")
        print(", ".join(missing_models))
        for name in missing_models:
            print(f"\nSearching best hyperparameters for {'\033[1m'}{name}{'\033[0m'}...")
            start = time.time()
            grid_search = GridSearchCV(
                estimator=models[name],
                param_grid=hyperparameters_grids[name],
                cv=5,
                scoring='f1_macro',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train_vector, y_train)
            elapsed = time.time() - start
            print(f"Took {elapsed:.2f} seconds.")
            print(f"Best hyperparameters for {name}: {grid_search.best_params_}")
            print(f"{'\033[92m'}Best f1_score: {grid_search.best_score_:.4f}{'\033[0m'}")

            saved_params[name] = grid_search.best_params_

    with open(param_file, 'w') as f:
        json.dump(saved_params, f, indent=4)
    print(f"\n{'\033[92m'}Saved all best hyperparameters to{'\033[0m'} {param_file}\n")

    for name, params in saved_params.items():
        print(f"{name} - Best hyperparameters: {'\033[95m'}{params}{'\033[0m'}")
        if name == "SVM":
            best_models[name] = LinearSVC(random_state=91, max_iter=5000, **params)
        elif name == "Naive Bayes":
            best_models[name] = MultinomialNB(**params)
        elif name == "Logistic Regression":
            best_models[name] = LogisticRegression(max_iter=1000, random_state=91, **params)
        elif name == "Random Forest":
            best_models[name] = RandomForestClassifier(random_state=91, n_jobs=-1, **params)

    return best_models
