"""Report the floating-point graph operation count for the GTSRB model.

This is the cost of TensorFlow's float simulation of the binary network; it is
not an XNOR/popcount hardware cost.
"""

import tensorflow as tf

from GTSRB_cbcnn import IMAGE_SHAPE, build_model


def get_flops(model, input_shape):
    concrete = tf.function(
        lambda inputs: model(inputs, training=False)
    ).get_concrete_function(
        tf.TensorSpec([1, *input_shape], tf.float32)
    )
    options = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    options["output"] = "none"
    profile = tf.compat.v1.profiler.profile(concrete.graph, options=options)
    return profile.total_float_ops


if __name__ == "__main__":
    model = build_model()
    print(get_flops(model, IMAGE_SHAPE))

