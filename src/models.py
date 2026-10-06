from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.regularizers import l2


def create_logistic_regression():
    return LogisticRegression(solver="lbfgs", max_iter=300)


def create_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        batch_size=128,
        learning_rate_init=0.001,
        max_iter=50,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
    )


def create_cnn(dropout_rate=0.5, l2_strength=0.0, augmentation=None):
    """Build the shared CNN, optionally adding augmentation and L2 regularization."""
    regularizer = l2(l2_strength) if l2_strength > 0 else None
    layers = [Input(shape=(28, 28, 1))]
    if augmentation is not None:
        layers.append(augmentation)

    layers.extend([
        Conv2D(32, (3, 3), activation="relu", kernel_regularizer=regularizer),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", kernel_regularizer=regularizer),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation="relu", kernel_regularizer=regularizer),
        Dropout(dropout_rate),
        Dense(10, activation="softmax"),
    ])
    return Sequential(layers)
