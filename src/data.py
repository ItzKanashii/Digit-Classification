from pathlib import Path

import numpy as np
from scipy.io import loadmat


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "emnist-digits.mat"


def load_mat_file(path=DATA_PATH):
    """
    Load the raw EMNIST MATLAB file.

    Returns
    -------
    dict
        MATLAB variables.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}"
        )

    return loadmat(path)


def load_emnist(path=DATA_PATH):
    """
    Load EMNIST Digits.

    Returns
    -------
    X_train : ndarray
        Training images, shape (n_train, 28, 28).

    y_train : ndarray
        Training labels, shape (n_train,).

    writers_train : ndarray
        Training writer IDs, shape (n_train,).

    X_test : ndarray
        Test images, shape (n_test, 28, 28).

    y_test : ndarray
        Test labels, shape (n_test,).

    writers_test : ndarray
        Test writer IDs, shape (n_test,).
    """

    data = load_mat_file(path)

    dataset = data["dataset"]

    train = dataset["train"][0, 0]
    test = dataset["test"][0, 0]

    X_train = train["images"][0, 0]
    y_train = train["labels"][0, 0]
    writers_train = train["writers"][0, 0]

    X_test = test["images"][0, 0]
    y_test = test["labels"][0, 0]
    writers_test = test["writers"][0, 0]

    y_train = y_train.ravel()
    writers_train = writers_train.ravel()

    y_test = y_test.ravel()
    writers_test = writers_test.ravel()
    X_train = X_train.reshape(-1, 28, 28).transpose(0, 2, 1)
    X_test = X_test.reshape(-1, 28, 28).transpose(0, 2, 1)

    return (
        X_train,
        y_train,
        writers_train,
        X_test,
        y_test,
        writers_test,
    )


def dataset_summary(
    X_train,
    y_train,
    writers_train,
    X_test,
    y_test,
    writers_test,
):
    """Print basic dataset statistics."""

    print("=" * 60)
    print("EMNIST DIGITS SUMMARY")
    print("=" * 60)

    print(f"Training images : {X_train.shape}")
    print(f"Training labels : {y_train.shape}")
    print(f"Test images     : {X_test.shape}")
    print(f"Test labels     : {y_test.shape}")

    print("\nPixel information")
    print("-" * 60)

    print(f"Dtype           : {X_train.dtype}")
    print(f"Minimum pixel   : {X_train.min()}")
    print(f"Maximum pixel   : {X_train.max()}")
    print(f"Mean pixel      : {X_train.mean():.4f}")
    print(f"Std pixel       : {X_train.std():.4f}")

    print("\nLabels")
    print("-" * 60)

    print(f"Classes         : {np.unique(y_train)}")
    print(f"Number classes  : {len(np.unique(y_train))}")

    print("\nWriters")
    print("-" * 60)

    print(f"Training writers: {len(np.unique(writers_train))}")
    print(f"Test writers    : {len(np.unique(writers_test))}")


if __name__ == "__main__":
    data = load_emnist()
    dataset_summary(*data)
