import numpy as np


def MAE(actual, pred):
    return np.mean(np.abs(actual - pred))


def MSE(actual, pred):
    return np.mean((actual - pred) ** 2)


def MAPE(actual, pred):
    return np.mean(np.abs((actual - pred) / actual))


def RMSE(actual, pred):
    return np.sqrt(MSE(actual, pred))


def MASE(actual, pred):
    n = actual.shape[0]
    d = np.abs(np.diff(actual)).sum() / (n - 1)
    errors = np.abs(actual, pred).sum()
    return errors / d
