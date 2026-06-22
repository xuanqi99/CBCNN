"""Binary Dense and Conv2D layers compatible with modern tf.keras."""

import math

import tensorflow as tf

from binary_ops import binarize


@tf.keras.utils.register_keras_serializable(package="CBCNN")
class Clip(tf.keras.constraints.Constraint):
    def __init__(self, min_value, max_value=None):
        max_value = -min_value if max_value is None else max_value
        self.min_value = min(min_value, max_value)
        self.max_value = max(min_value, max_value)

    def __call__(self, weights):
        return tf.clip_by_value(weights, self.min_value, self.max_value)

    def get_config(self):
        return {"min_value": self.min_value, "max_value": self.max_value}


def _resolve_h(H, fan_in, fan_out):
    if H == "Glorot":
        return math.sqrt(1.5 / (fan_in + fan_out))
    value = float(H)
    if value <= 0:
        raise ValueError("H must be positive or 'Glorot'.")
    return value


@tf.keras.utils.register_keras_serializable(package="CBCNN")
class BinaryDense(tf.keras.layers.Dense):
    """Dense layer whose latent kernel is binarized during every forward pass."""

    def __init__(self, units, H=1.0, **kwargs):
        self.H = H
        self._binary_H = None
        super().__init__(units, **kwargs)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        self._binary_H = _resolve_h(self.H, input_dim, self.units)
        self.kernel_initializer = tf.keras.initializers.RandomUniform(
            -self._binary_H, self._binary_H
        )
        self.kernel_constraint = Clip(-self._binary_H, self._binary_H)
        super().build(input_shape)

    def call(self, inputs):
        outputs = tf.linalg.matmul(inputs, binarize(self.kernel, self._binary_H))
        if self.bias is not None:
            outputs = tf.nn.bias_add(outputs, self.bias)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"H": self.H})
        return config


@tf.keras.utils.register_keras_serializable(package="CBCNN")
class BinaryConv2D(tf.keras.layers.Conv2D):
    """Conv2D layer whose latent kernel is binarized during every forward pass."""

    def __init__(self, filters, kernel_size, H=1.0, **kwargs):
        self.H = H
        self._binary_H = None
        super().__init__(filters, kernel_size, **kwargs)

    def build(self, input_shape):
        channel_axis = 1 if self.data_format == "channels_first" else -1
        input_dim = input_shape[channel_axis]
        if input_dim is None:
            raise ValueError("The input channel dimension must be defined.")

        kernel_area = self.kernel_size[0] * self.kernel_size[1]
        fan_in = int(input_dim) * kernel_area
        fan_out = self.filters * kernel_area
        self._binary_H = _resolve_h(self.H, fan_in, fan_out)
        self.kernel_initializer = tf.keras.initializers.RandomUniform(
            -self._binary_H, self._binary_H
        )
        self.kernel_constraint = Clip(-self._binary_H, self._binary_H)
        super().build(input_shape)

    def call(self, inputs):
        outputs = self.convolution_op(
            inputs, binarize(self.kernel, self._binary_H)
        )
        if self.bias is not None:
            data_format = "NCHW" if self.data_format == "channels_first" else "NHWC"
            outputs = tf.nn.bias_add(outputs, self.bias, data_format=data_format)
        if self.activation is not None:
            outputs = self.activation(outputs)
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update({"H": self.H})
        return config


BinaryConvolution2D = BinaryConv2D

