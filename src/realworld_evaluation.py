"""Load, evaluate, and visualize labeled real-world handwritten digit surveys.

The expected NPZ fields are ``X_test``, ``Y_test``, and ``writer`` (the schema
used by the example survey archive). ``images``/``labels``/``writer_name``
are accepted as aliases. Images may be NHW, NHWC, or NCHW, at 28x28 resolution.

Typical notebook use from ``notebooks/11_test.ipynb``::

    from src.realworld_evaluation import evaluate_survey_file

    results = evaluate_survey_file(
        "../data/survey/realworld_dataset.npz",
        "../models/cnn_writer_independent.keras",
        "../outputs/realworld",
    )

The function returns predictions and per-writer metrics, and writes one
prediction sheet and one score chart PNG for each writer.
"""

from pathlib import Path
import re

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont


def _find_array(archive, names, description):
    for name in names:
        if name in archive.files:
            return archive[name]
    raise KeyError(
        f"Could not find {description} in NPZ. Expected one of: {', '.join(names)}. "
        f"Available fields: {', '.join(archive.files)}"
    )


def prepare_survey_images(images):
    """Convert survey images to float32 ``(N, 28, 28, 1)`` in the 0–1 range.

    The image orientation and foreground/background convention are preserved;
    this function only handles channel layout and pixel scaling.
    """
    values = np.asarray(images)
    if values.ndim == 2:
        values = values[np.newaxis, ...]

    if values.ndim == 4:
        if values.shape[1:] == (1, 28, 28):  # NCHW
            values = values[:, 0, :, :]
        elif values.shape[1:] == (28, 28, 1):  # NHWC
            values = values[:, :, :, 0]
        else:
            raise ValueError(
                "Four-dimensional survey images must have shape "
                "(N, 1, 28, 28) or (N, 28, 28, 1); "
                f"received {values.shape}."
            )
    elif values.ndim == 3:
        if values.shape[1:] != (28, 28):
            raise ValueError(
                f"Three-dimensional images must have shape (N, 28, 28); received {values.shape}."
            )
    else:
        raise ValueError(
            "Survey images must be a 28x28 image batch with 1 grayscale channel; "
            f"received {values.shape}."
        )

    if not np.issubdtype(values.dtype, np.number):
        raise TypeError(f"Image values must be numeric, received {values.dtype}.")
    values = values.astype(np.float32)
    if values.size == 0:
        raise ValueError("The survey archive contains no images.")
    if not np.isfinite(values).all():
        raise ValueError("Survey images contain NaN or infinite pixel values.")
    minimum = float(values.min())
    maximum = float(values.max())
    if minimum < 0:
        raise ValueError(f"Pixel values must be nonnegative; minimum is {minimum}.")
    if maximum > 255:
        raise ValueError(f"Pixel values must be in [0, 1] or [0, 255]; maximum is {maximum}.")
    if maximum > 1:
        values /= 255.0
    return values[..., np.newaxis]


def load_survey_npz(npz_path):
    """Load images, true digit labels, and writer names from a survey NPZ.

    Supports the provided schema (``X_test``, ``Y_test``, ``writer``) and the
    aliases ``images``, ``labels``, and ``writer_name``. Object/pickled arrays
    are deliberately not enabled.
    """
    npz_path = Path(npz_path)
    if not npz_path.is_file():
        raise FileNotFoundError(f"Survey NPZ not found: {npz_path}")

    with np.load(npz_path, allow_pickle=False) as archive:
        images = _find_array(archive, ("X_test", "images", "X"), "image array")
        labels = _find_array(archive, ("Y_test", "labels", "y"), "digit labels")
        writers = _find_array(
            archive,
            ("writer", "writer_name", "writer_names"),
            "writer names",
        )

    images = prepare_survey_images(images)
    labels = np.asarray(labels).reshape(-1)
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.issubdtype(labels.dtype, np.number) or not np.all(labels == np.floor(labels)):
            raise ValueError("Labels must be integer digits from 0 through 9.")
    labels = labels.astype(np.int64)
    if np.any((labels < 0) | (labels > 9)):
        raise ValueError("Labels must be digits from 0 through 9.")

    writers = np.asarray(writers).reshape(-1)
    if writers.size == 1 and len(labels) > 1:
        writers = np.repeat(writers, len(labels))
    if len(images) != len(labels) or len(images) != len(writers):
        raise ValueError(
            "Image, label, and writer counts must match; got "
            f"{len(images)}, {len(labels)}, and {len(writers)}."
        )
    if writers.dtype.kind not in "USiu":
        raise TypeError(
            "Writer values must be strings or integers; object arrays are not supported."
        )
    writers = np.asarray([str(value).strip() for value in writers])
    if np.any(writers == ""):
        raise ValueError("Writer names must not be empty.")
    return images, labels, writers


def _metrics_for_indices(labels, predictions, indices):
    true = labels[indices]
    predicted = predictions[indices]
    per_digit = {}
    recalls = []
    for digit in range(10):
        mask = true == digit
        support = int(mask.sum())
        correct = int(np.sum(predicted[mask] == digit)) if support else 0
        recall = correct / support if support else None
        per_digit[digit] = {
            "support": support,
            "correct": correct,
            "wrong": support - correct,
            "recall": recall,
        }
        if recall is not None:
            recalls.append(recall)
    correct_count = int(np.sum(true == predicted))
    total = len(indices)
    return {
        "total": total,
        "correct": correct_count,
        "wrong": total - correct_count,
        "accuracy": correct_count / total if total else 0.0,
        "macro_recall": float(np.mean(recalls)) if recalls else 0.0,
        "per_digit": per_digit,
    }


def predict_survey_images(images, labels, writers, model, batch_size=128):
    """Predict survey images and calculate accuracy/recall metrics by writer."""
    images = prepare_survey_images(images)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    writers = np.asarray([str(value).strip() for value in np.asarray(writers).reshape(-1)])
    if len(images) != len(labels) or len(images) != len(writers):
        raise ValueError("Images, labels, and writer names must have matching lengths.")
    if len(labels) == 0:
        raise ValueError("No survey examples were provided.")
    if np.any((labels < 0) | (labels > 9)):
        raise ValueError("Labels must be digits from 0 through 9.")

    probabilities = model.predict(images, batch_size=batch_size, verbose=0)
    if probabilities.shape != (len(images), 10):
        raise ValueError(
            "Expected model output with 10 digit probabilities per image; "
            f"received {probabilities.shape}."
        )
    predictions = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)
    unique_writers = list(dict.fromkeys(writers.tolist()))
    writer_metrics = {}
    for writer in unique_writers:
        indices = np.flatnonzero(writers == writer)
        writer_metrics[writer] = _metrics_for_indices(labels, predictions, indices)
    return {
        "images": images,
        "labels": labels,
        "writers": writers,
        "predictions": predictions,
        "probabilities": probabilities,
        "confidence": confidence,
        "writer_metrics": writer_metrics,
    }


def _safe_filename(value):
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._-")
    return slug or "writer"


def _load_report_font(size):
    """Load a readable TrueType font, with Pillow's built-in font as fallback."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def save_writer_prediction_sheets(
    results,
    output_dir,
    columns=10,
    pixel_scale=10,
    horizontal_gap=10,
    caption_height=30,
    header_height=200,
    bottom_margin=10,
    side_margin=10,
    canvas_scale=1,
):
    """Save one exact-pixel PNG per writer, with column-wise image ordering.

    ``canvas_scale`` multiplies all canvas dimensions, image pixels, text, gaps,
    and margins. With 100 images and the current 200px header, scale 1 yields
    2910x3310 pixels; scale 2 yields 5820x6620 pixels.
    """
    if not isinstance(canvas_scale, (int, np.integer)) or canvas_scale < 1:
        raise ValueError("canvas_scale must be a positive integer.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    writers = results["writers"]
    tile_size = 28 * pixel_scale * canvas_scale
    scaled_gap = horizontal_gap * canvas_scale
    scaled_caption_height = caption_height * canvas_scale
    scaled_header_height = header_height * canvas_scale
    scaled_bottom_margin = bottom_margin * canvas_scale
    scaled_side_margin = side_margin * canvas_scale
    title_font = _load_report_font(32 * canvas_scale)
    detail_font = _load_report_font(20 * canvas_scale)
    caption_font = _load_report_font(15 * canvas_scale)

    for writer, metrics in results["writer_metrics"].items():
        indices = np.flatnonzero(writers == writer)
        rows = int(np.ceil(len(indices) / columns))
        width = (
            scaled_side_margin * 2
            + columns * tile_size
            + (columns - 1) * scaled_gap
        )
        height = (
            scaled_header_height
            + rows * (tile_size + scaled_caption_height)
            + scaled_bottom_margin
        )
        sheet = Image.new("RGB", (width, height), color="#1e2121")
        draw = ImageDraw.Draw(sheet)
        header_title = f"Survey predictions — Writer {writer}"
        header_details = (
            f"{metrics['correct']}/{metrics['total']} correct    "
            f"Accuracy: {metrics['accuracy']:.1%}    "
            f"Macro recall: {metrics['macro_recall']:.1%}"
        )
        title_box = draw.textbbox((0, 0), header_title, font=title_font)
        details_box = draw.textbbox((0, 0), header_details, font=detail_font)
        title_x = max(0, (width - (title_box[2] - title_box[0])) // 2)
        details_x = max(0, (width - (details_box[2] - details_box[0])) // 2)
        draw.text((title_x, 8 * canvas_scale), header_title, fill="white", font=title_font)
        draw.text((details_x, 53 * canvas_scale), header_details, fill="white", font=detail_font)

        # Fill top-to-bottom within each column, then continue to the next column.
        for position, idx in enumerate(indices):
            row = position % rows
            column = position // rows
            x = scaled_side_margin + column * (tile_size + scaled_gap)
            y = scaled_header_height + row * (tile_size + scaled_caption_height)

            image = results["images"][idx].squeeze()
            pixels = np.rint(np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
            tile = Image.fromarray(pixels).resize(
                (tile_size, tile_size),
                resample=Image.Resampling.NEAREST,
            )
            sheet.paste(tile.convert("RGB"), (x, y))

            correct = results["labels"][idx] == results["predictions"][idx]
            color = "#11FF11" if correct else "#FF1111"
            draw.rectangle(
                (x, y, x + tile_size - 1, y + tile_size - 1),
                outline=color,
                width=2 * canvas_scale,
            )
            draw.text(
                (x, y + tile_size + 5 * canvas_scale),
                f"Pred: {results['predictions'][idx]} | "
                f"True: {results['labels'][idx]} | "
                f"Confidence: {results['confidence'][idx]:.2f}",
                fill=color,
                font=caption_font,
            )

        path = output_dir / f"{_safe_filename(writer)}_prediction.png"
        sheet.save(path, format="PNG", optimize=True)
        saved[writer] = str(path)
    return saved


def save_writer_metric_charts(results, output_dir):
    """Save one PNG per writer with accuracy, recall, and correct/wrong counts."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for writer, metrics in results["writer_metrics"].items():
        digits = [digit for digit, value in metrics["per_digit"].items() if value["support"] > 0]
        recalls = [metrics["per_digit"][digit]["recall"] for digit in digits]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.3))

        score_names = ["Accuracy", "Macro recall"]
        score_values = [metrics["accuracy"], metrics["macro_recall"]]
        score_bars = axes[0].bar(score_names, score_values, color=["#4C78A8", "#F58518"])
        axes[0].set_ylim(0, 1.08)
        axes[0].set_ylabel("Score")
        axes[0].set_title("Overall performance")
        axes[0].tick_params(axis="x", rotation=15)
        axes[0].grid(axis="y", alpha=0.25)
        for bar, value in zip(score_bars, score_values):
            axes[0].text(bar.get_x() + bar.get_width()/2, value + 0.02,
                         f"{value:.1%}", ha="center", fontsize=9)

        count_names = ["Correct", "Wrong"]
        count_values = [metrics["correct"], metrics["wrong"]]
        count_bars = axes[1].bar(count_names, count_values, color=["#2DA44E", "#D1242F"])
        axes[1].set_title(f"Predictions (n={metrics['total']})")
        axes[1].set_ylabel("Images")
        axes[1].grid(axis="y", alpha=0.25)
        for bar, value in zip(count_bars, count_values):
            axes[1].text(bar.get_x() + bar.get_width()/2, value,
                         str(value), ha="center", va="bottom", fontsize=9)

        axes[2].bar([str(digit) for digit in digits], recalls, color="#6F42C1")
        axes[2].set_ylim(0, 1.08)
        axes[2].set_title("Recall by true digit")
        axes[2].set_xlabel("Digit")
        axes[2].set_ylabel("Recall")
        axes[2].grid(axis="y", alpha=0.25)
        for x, value in enumerate(recalls):
            axes[2].text(x, value + 0.02, f"{value:.0%}", ha="center", fontsize=8)

        fig.suptitle(f"Survey results — writer {writer}\nMacro recall averages digits present in this writer's labels")
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        path = output_dir / f"{_safe_filename(writer)}_metrics.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        saved[writer] = str(path)
    return saved


def evaluate_survey_file(
    npz_path, model_path, output_dir, batch_size=128, prediction_scale=1
):
    """Load a survey NPZ and saved Keras model, evaluate, and save per-writer PNGs.

    Returns a dictionary containing per-image predictions, per-writer metrics,
    and the paths to prediction sheets and metric charts. ``prediction_scale``
    is an integer multiplier for prediction sheet size and rendering.
    """
    import tensorflow as tf

    images, labels, writers = load_survey_npz(npz_path)
    model = tf.keras.models.load_model(model_path)
    results = predict_survey_images(images, labels, writers, model, batch_size=batch_size)
    output_dir = Path(output_dir)
    prediction_sheets = save_writer_prediction_sheets(
        results,
        output_dir,
        canvas_scale=prediction_scale,
    )
    metric_charts = save_writer_metric_charts(results, output_dir)
    results["prediction_sheets"] = prediction_sheets
    results["metric_charts"] = metric_charts
    results["output_dir"] = str(output_dir)
    for writer, metrics in results["writer_metrics"].items():
        print(
            f"{writer}: accuracy={metrics['accuracy']:.2%}, "
            f"macro recall={metrics['macro_recall']:.2%}, "
            f"correct={metrics['correct']}, wrong={metrics['wrong']}"
        )
    return results
