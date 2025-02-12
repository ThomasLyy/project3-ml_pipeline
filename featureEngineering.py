import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def feature_engineer():
    """Interactively guides the user through feature engineering tasks."""

    # Load data with error handling
    while True:
        try:
            filepath = input("Enter the path to your .csv file: ")
            df = pd.read_csv(filepath)
            print(df.head().to_markdown(index=False, numalign="left", stralign="left"))
            break
        except FileNotFoundError:
            print("File not found. Please check the path and try again.")

    # Feature relationship analysis (optional)
    if input("Analyze feature relationships? (yes/no): ").lower() == "yes":
        sns.pairplot(df)
        plt.show()

        while True:
            specific_features = input(
                "Enter specific features to analyze (comma-separated, or 'all'): "
            )
            if specific_features.lower() == "all":
                sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
                plt.show()
                break
            else:
                try:
                    features_list = [f.strip() for f in specific_features.split(",")]
                    sns.pairplot(df[features_list])
                    plt.show()
                    break
                except KeyError:
                    print("Invalid feature names. Please check and try again.")

    # Handle missing data with customizable options
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            print(f"\nMissing values in '{col}': {df[col].isnull().sum()}")
            while True:
                fill_method = input(
                    f"How to fill missing values in '{col}'? (mean, median, mode, drop, constant, custom): "
                )

                if fill_method.lower() == "mean":
                    df[col] = df[col].fillna(df[col].mean())
                    break
                elif fill_method.lower() == "median":
                    df[col] = df[col].fillna(df[col].median())
                    break
                elif fill_method.lower() == "mode":
                    df[col] = df[col].fillna(df[col].mode()[0])
                    break
                elif fill_method.lower() == "drop":
                    df = df.dropna(subset=[col])
                    break
                elif fill_method.lower() == "constant":
                    fill_value = input("Enter the constant value to fill with: ")
                    df[col] = df[col].fillna(fill_value)
                    break
                elif fill_method.lower() == "custom":
                    try:
                        # Let the user enter a custom Python expression to fill
                        fill_expr = input(
                            "Enter a Python expression to calculate the fill value (e.g., 'df['col2'].mean()'): "
                        )
                        df[col] = df[col].fillna(eval(fill_expr))
                        break
                    except Exception as e:
                        print(f"Error evaluating expression: {e}. Please try again.")
                else:
                    print(
                        "Invalid fill method. Please choose from the available options."
                    )

    # Create new features with guided input and error handling
    while input("Create new features? (yes/no): ").lower() == "yes":
        new_feature_name = input("Enter name for new feature: ")
        while True:
            new_feature_expr = input(
                "Enter expression for new feature (e.g., 'col1 * col2'): "
            )
            try:
                df[new_feature_name] = df.eval(new_feature_expr)
                break  # Exit the inner loop if successful
            except Exception as e:
                print(
                    f"Error creating feature: {e}. Please check your expression and try again."
                )

    # Display final dataset
    print("\nFinal dataset:")
    print(df.head().to_markdown(index=False, numalign="left", stralign="left"))


if __name__ == "__main__":
    feature_engineer()
