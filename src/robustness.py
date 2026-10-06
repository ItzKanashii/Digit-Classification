import numpy as np
import tensorflow as tf


def evaluate_condition(model, X_test, y_test, condition, *, batch_size=512, seed=0):
    correct_count = 0
    total_loss = 0.0
    total_count = 0
    rng = np.random.default_rng(seed)

    for start in range(0, len(X_test), batch_size):
        stop = min(start + batch_size, len(X_test))
        images = tf.convert_to_tensor(X_test[start:stop], dtype=tf.float32)
        labels = y_test[start:stop]

        if condition is None:
            batch = images
        elif isinstance(condition, tuple) and condition[0] == "noise":
            _, sigma, _ = condition
            noise = rng.normal(0.0, sigma, size=images.shape).astype(np.float32)
            batch = tf.clip_by_value(images + noise, 0.0, 1.0)
        else:
            batch = condition(images, training=True)

        probabilities = model(batch, training=False).numpy()
        predictions = np.argmax(probabilities, axis=1)
        correct_count += int(np.sum(predictions == labels))
        chosen = np.clip(probabilities[np.arange(len(labels)), labels], 1e-7, 1.0)
        total_loss += float(np.sum(-np.log(chosen)))
        total_count += len(labels)

    return {
        "accuracy": correct_count / total_count,
        "loss": total_loss / total_count,
        "correct": correct_count,
        "total": total_count,
    }
