import numpy as np


def to_one_hot(x: int, d: int) -> np.ndarray:
    """
    Convert an integer to a one-hot vector of dimension d
    """
    v = np.zeros(d)
    v[x] = 1.0
    return v


def from_one_hot(v: np.ndarray) -> int:
    """
    Convert a one-hot vector to an integer
    """
    return np.argmax(v)
