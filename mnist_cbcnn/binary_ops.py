"""Straight-through binary operators used by the CBCNN layers."""

import tensorflow as tf


@tf.keras.utils.register_keras_serializable(package="CBCNN")
def round_through(x):
    """Round in the forward pass and use the identity STE backward."""
    rounded = tf.round(x)
    return x + tf.stop_gradient(rounded - x)


def _hard_sigmoid(x):
    return tf.clip_by_value(0.5 * x + 0.5, 0.0, 1.0)


@tf.keras.utils.register_keras_serializable(package="CBCNN")
def binary_tanh(x):
    """Return -1/+1 with a clipped straight-through gradient."""
    return 2.0 * round_through(_hard_sigmoid(x)) - 1.0


@tf.keras.utils.register_keras_serializable(package="CBCNN")
def binarize(weights, H=1.0):
    """Binarize weights to {-H, +H}."""
    H = tf.cast(H, weights.dtype)
    return H * binary_tanh(weights / H)
