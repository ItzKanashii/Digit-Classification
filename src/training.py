import tensorflow as tf


def compile_model(model, optimizer="adam", learning_rate=0.001):
    """Compile a classifier with the project's standard sparse-label objective."""
    if isinstance(optimizer, str):
        if optimizer.lower() == "adam":
            optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        elif optimizer.lower() == "sgd":
            optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer}")

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def create_early_stopping(patience=2, monitor="val_loss"):
    """Create early stopping that restores the best validation weights."""
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor,
        patience=patience,
        restore_best_weights=True,
    )


def fit_and_evaluate(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    *,
    epochs=15,
    batch_size=128,
    validation_split=0.1,
    callbacks=None,
    verbose=1,
):
    """Fit a compiled model and return history and test metrics."""
    history = model.fit(
        X_train,
        y_train,
        validation_split=validation_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose,
    )
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    return {
        "model": model,
        "history": history,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }
