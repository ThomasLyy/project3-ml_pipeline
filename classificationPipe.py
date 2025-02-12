import os  # For interacting with the operating system (file paths, etc.)

import pandas as pd  # For working with dataframes (tabular data)
import pandas.api.types as ptypes  # For checking data types within pandas series
from sklearn.ensemble import RandomForestClassifier  # Random forest model
from sklearn.linear_model import LogisticRegression  # Logistic regression model
from sklearn.metrics import accuracy_score  # Model evaluation metrics

# Machine learning libraries (scikit-learn)
from sklearn.model_selection import (  # Splitting data, hyperparameter tuning, model evaluation
    GridSearchCV,
    cross_val_score,
    train_test_split,
)
from sklearn.naive_bayes import GaussianNB  # Naive Bayes model
from sklearn.neighbors import KNeighborsClassifier  # K-nearest neighbors model
from sklearn.preprocessing import (  # Feature scaling and encoding
    OneHotEncoder,
    StandardScaler,
)
from sklearn.svm import SVC  # Support Vector Machine model
from sklearn.tree import DecisionTreeClassifier  # Decision tree model

# Utility libraries
from tqdm import tqdm  # Progress bar for long-running tasks


def main():
    # Input Paths with error handling
    while True:
        train_path = input("Enter the path to your training dataset (CSV): ")
        if not train_path:  # Handle empty input
            print("Training dataset path is required. Please enter a valid path.")
            continue
        try:
            if not os.path.exists(train_path):
                raise FileNotFoundError
            train_data = pd.read_csv(train_path)
            break
        except FileNotFoundError:
            print("File not found. Please check the path and try again.")
        except OSError as e:
            if e.errno == 22:
                print("Invalid argument in file path. Check the formatting.")
            else:
                raise e

    # Load Training Data
    train_data = pd.read_csv(train_path)
    print("\nTraining Data Head:\n")
    print(train_data.head().to_markdown(index=False, numalign="left", stralign="left"))

    # Target Variable Identification (by number)
    while True:
        column_list = ", ".join(
            [f"{i+1}. {col}" for i, col in enumerate(train_data.columns)]
        )
        print(f"\nColumns in your dataset: {column_list}")

        try:
            choice = int(
                input("\nEnter the number of the column you want to classify: ")
            )
            if 1 <= choice <= len(train_data.columns):
                target_column = train_data.columns[choice - 1]  # Convert to string here
                break
            else:
                print("Invalid choice. Please enter a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")

    # Missing Value Handling (individual choice per column)
    missing_cols = train_data.columns[train_data.isnull().any()].tolist()
    if missing_cols:
        print("\nColumns with missing values:", missing_cols)

        # Print options once before the loop (compact format)
        imputation_options = ["median", "mode", "drop", "other"]
        option_list = ", ".join(
            [f"{i+1}. {method}" for i, method in enumerate(imputation_options)]
        )
        print(f"\nOptions for imputing missing values: {option_list}")

        for col in missing_cols:
            while True:
                try:
                    choice = int(
                        input(f"Enter the number of the method to use for '{col}': ")
                    )
                    if 1 <= choice <= len(imputation_options):
                        impute_method = imputation_options[choice - 1]
                        if impute_method == "other":
                            # Get custom method if "other" is chosen
                            while True:
                                impute_method = input(
                                    "Enter the method ('median', 'mode', 'drop', or other): "
                                )
                                if impute_method in ["median", "mode", "drop"]:
                                    break
                                else:
                                    print(
                                        "Invalid input. Please choose from 'median', 'mode', 'drop', or other."
                                    )
                        break  # Exit the inner while loop after getting a valid input
                    else:
                        print("Invalid choice. Please enter a number from the list.")
                except ValueError:
                    print("Invalid input. Please enter a number.")

            # Apply the chosen imputation method
            if impute_method == "median":
                if ptypes.is_numeric_dtype(train_data[col]):
                    train_data[col] = train_data[col].fillna(train_data[col].median())
                else:
                    print(f"Column '{col}' is not numeric. Cannot impute with median.")
            elif impute_method == "mode":
                mode_value = train_data[col].mode()[0]
                train_data[col] = train_data[col].fillna(mode_value)
            elif impute_method == "drop":
                train_data.drop(col, axis=1, inplace=True)

    # Drop Unnecessary Columns (by number)
    while True:
        drop_input = input(
            "\nEnter the numbers of the columns you want to drop (comma-separated), or press Enter to keep all: "
        )
        if not drop_input:  # Empty input means keep all columns
            break
        try:
            drop_choices = [int(x.strip()) for x in drop_input.split(",")]
            valid_choices = all(1 <= c <= len(train_data.columns) for c in drop_choices)
            if valid_choices:
                drop_columns = [train_data.columns[c - 1] for c in drop_choices]
                train_data.drop(drop_columns, axis=1, inplace=True)
                break
            else:
                print(
                    "Invalid choices. Please enter numbers from the list, separated by commas."
                )
        except ValueError:
            print("Invalid input. Please enter numbers separated by commas.")

    # One-Hot Encoding for Categorical Features
    categorical_cols = train_data.select_dtypes(include=["object"]).columns.tolist()
    if categorical_cols:
        print("\nCategorical columns detected:", categorical_cols)
        print("One-hot encoding columns.\n")
        onehot_encoder = OneHotEncoder(drop="first")
        train_data_encoded = onehot_encoder.fit_transform(
            train_data[categorical_cols]
        ).toarray()
        encoded_df = pd.DataFrame(
            train_data_encoded,
            columns=onehot_encoder.get_feature_names_out(categorical_cols),
        )
        train_data = pd.concat(
            [train_data.drop(categorical_cols, axis=1), encoded_df], axis=1
        )

    # Data Preparation
    X = train_data.drop(target_column, axis=1)
    y = train_data[target_column]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=5
    )

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Model Selection and Hyperparameter Tuning
    models = {
        "LogisticRegression": (
            LogisticRegression(),
            {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
        ),
        "RandomForestClassifier": (
            RandomForestClassifier(),
            {"n_estimators": [10, 50, 100, 200], "max_depth": [None, 5, 10, 15]},
        ),
        "KNeighborsClassifier": (KNeighborsClassifier(), {"n_neighbors": [3, 5, 7, 9]}),
        "SVC": (SVC(), {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}),
        "GaussianNB": (GaussianNB(), {}),  # No hyperparameters to tune
        "DecisionTreeClassifier": (
            DecisionTreeClassifier(),
            {"max_depth": [None, 5, 10, 15]},
        ),
    }

    # Training iteration of each model with a Progress Bar
    results_list = []

    for modelName, (model, paramGrid) in tqdm(models.items(), desc="Training Models"):
        gridSearch = GridSearchCV(
            estimator=model, param_grid=paramGrid, cv=5, scoring="accuracy"
        )
        # Get the Cross-Validation Splitter or Number of Splits
        cv_value = gridSearch.cv
        if isinstance(cv_value, int):
            n_splits = cv_value
        else:
            n_splits = cv_value.get_n_splits(X_train_scaled, y_train)
        # Ensure Integer Total for tqdm
        total_iterations = n_splits * len(paramGrid)
        total_iterations = (
            int(total_iterations)
            if total_iterations == int(total_iterations)
            else round(total_iterations)
        )
        # Nested Progress Bar for GridSearchCV (with corrected total)
        with tqdm(
            total=total_iterations, desc=f"Tuning {modelName}", leave=False
        ) as pbar:

            def update_progress(self, *args):
                pbar.update(1)

            gridSearch._check_is_fitted = update_progress
            gridSearch.fit(X_train_scaled, y_train)
            y_pred = gridSearch.predict(X_val_scaled)
            accuracy = accuracy_score(y_val, y_pred)
            scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
            results_list.append(
                {
                    "Modèle": modelName,
                    "Accuracy": accuracy,
                    "Mean Accuracy": gridSearch.best_score_,
                    "Cross-Validation": scores,
                    "Mean Cross-Validation": scores.mean(),
                }
            )

    # Create a DataFrame from the list of dictionaries
    results_df = pd.DataFrame(results_list)

    # Sort results by mean accuracy (descending)
    results_df = results_df.sort_values(by="Mean Accuracy", ascending=False)

    # Print the results
    print(results_df.to_markdown(index=False, numalign="left", stralign="left"))

    # Identify and print the best and worst performing algorithms
    best_model = results_df.loc[results_df["Mean Accuracy"].idxmax(), "Modèle"]
    worst_model = results_df.loc[results_df["Mean Accuracy"].idxmin(), "Modèle"]
    best_accuracy = results_df["Mean Accuracy"].max()
    worst_accuracy = results_df["Mean Accuracy"].min()

    print(f"Best performing model: {best_model} with accuracy {best_accuracy:.3f}")
    print(f"Worst performing model: {worst_model} with accuracy {worst_accuracy:.3f}")


# CLI Call
if __name__ == "__main__":
    main()
