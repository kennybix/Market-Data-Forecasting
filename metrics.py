# metrics.py
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculate standard regression metrics.
    
    Returns:
        mse: Mean Squared Error.
        r2: R^2 Score.
    """
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, r2

def calculate_directional_accuracy(y_true, y_pred):
    """
    Calculates directional accuracy.
    
    Both y_true and y_pred are continuous values. This function converts
    them to directions (1 if >= 0, else -1) and computes the proportion of
    times the predicted direction matches the true direction.
    """
    y_true_dir = np.where(y_true >= 0, 1, -1)
    y_pred_dir = np.where(y_pred >= 0, 1, -1)
    directional_accuracy = np.mean(y_true_dir == y_pred_dir)
    return directional_accuracy

def measure_performance(y_true, y_pred):
    """
    Wrapper that computes a suite of regression and directional metrics.
    
    Returns:
        A dictionary with keys 'MSE', 'R2', and 'Directional_Accuracy'.
    """
    mse, r2 = calculate_regression_metrics(y_true, y_pred)
    direction_acc = calculate_directional_accuracy(y_true, y_pred)
    return {
        'MSE': mse,
        'R2': r2,
        'Directional_Accuracy': direction_acc
    }
