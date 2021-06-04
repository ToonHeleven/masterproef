import numpy as np


def evaluate(predictions: np.ndarray, labels: np.ndarray, padfirst=0):
    isinarray = [(labels[i] in prediction) for i, prediction in enumerate(predictions)]
    isinarray = padfirst*[False] + isinarray
    return isinarray
