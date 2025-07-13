import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from metrics import measure_performance


def train_and_forecast(x_train_path="data/x_train.csv",
                       y_train_path="data/y_train.csv",
                       x_test_path="data/x_test.csv",
                       y_test_path="data/y_test.csv"):
    """Train RandomForest models and forecast test set."""
    # Load datasets
    x_train = pd.read_csv(x_train_path, index_col="timestamp")
    y_train = pd.read_csv(y_train_path, index_col="timestamp")
    x_test = pd.read_csv(x_test_path, index_col="timestamp")
    y_test = pd.read_csv(y_test_path, index_col="timestamp")

    # Prepare container for predictions
    preds = pd.DataFrame(index=y_test.index, columns=y_test.columns)

    # Fit a model per target column
    for target in y_train.columns:
        model = RandomForestRegressor(n_estimators=200, random_state=42)
        model.fit(x_train, y_train[target])
        preds[target] = model.predict(x_test)

    # Evaluate
    metrics = measure_performance(y_test.values, preds.values)
    return preds, metrics


if __name__ == "__main__":
    predictions, evaluation = train_and_forecast()
    print("Predictions:\n", predictions.head())
    print("\nEvaluation metrics:\n", evaluation)
