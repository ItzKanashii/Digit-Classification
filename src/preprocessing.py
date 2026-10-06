import numpy as np


def normalize_images(X):
    """
    Scale pixel values from [0, 255] to [0, 1].
    """
    return X.astype(np.float32) / 255.0


def add_channel_dimension(X):
    """
    Convert (N, 28, 28) -> (N, 28, 28, 1).
    """
    return X[..., np.newaxis]


def preprocess_images(X):
    """
    Complete image preprocessing pipeline.
    """
    X = normalize_images(X)
    X = add_channel_dimension(X)

    return X
