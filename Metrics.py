# Import required dependencies
import numpy as np



def MAE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Mean Absolute Error (MAE) """
    # (1/n) * Σ|y_true - y_pred|
    return np.mean(np.abs(y_true - y_pred))


def MSE(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Mean Squared Error (MSE) """
    # (1/n) * Σ(y_true - y_pred)²
    return np.mean((y_true - y_pred) ** 2)


def R2(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the R2 metric """
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    # 1 - (SS_res / SS_tot)
    return 1 - ss_res / ss_tot


def Corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.float64:
    """ Implementation of the Pearson's Correlation Coefficient """
    # cov(y_true, y_pred) / (σ_y_true * σ_y_pred)
    return np.corrcoef(y_true, y_pred)[0, 1]