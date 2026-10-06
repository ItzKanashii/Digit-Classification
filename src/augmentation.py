import tensorflow as tf
from tensorflow.keras import Sequential


def create_augmentation(
    rotation_factor=0.08,
    translation_height=0.1,
    translation_width=0.1,
    zoom_factor=0.1
):
    """
    Create a data augmentation pipeline for 28x28 grayscale images.

    Parameters
    ----------
    rotation_factor : float
        Fraction of a full rotation used for random rotation.
    translation_height : float
        Maximum vertical translation as a fraction of image height.
    translation_width : float
        Maximum horizontal translation as a fraction of image width.
    zoom_factor : float
        Maximum random zoom range.

    Returns
    -------
    tf.keras.Sequential
        Keras augmentation pipeline.
    """

    return Sequential([
        tf.keras.layers.RandomRotation(rotation_factor),
        tf.keras.layers.RandomTranslation(
            height_factor=translation_height,
            width_factor=translation_width
        ),
        tf.keras.layers.RandomZoom(zoom_factor)
    ])
