"""Train and evaluate a binary CNN on MNIST."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from binary_layers import BinaryConv2D, BinaryDense
from binary_ops import binary_tanh

SEED = 37
IMAGE_SHAPE = (28, 28, 1)
NUM_CLASSES = 10
H = 1.0
LR_START = 1e-3
LR_END = 1e-4
EPSILON = 1e-6
MOMENTUM = 0.9
OUTPUT_DIR = Path(__file__).resolve().parent


def load_data():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis].astype("float32") / 255.0
    x_test = x_test[..., np.newaxis].astype("float32") / 255.0
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES) * 2.0 - 1.0
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES) * 2.0 - 1.0
    return (x_train, y_train), (x_test, y_test)


def build_model():
    return tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=IMAGE_SHAPE),
            BinaryConv2D(128, 3, H=H, padding="same", use_bias=False),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.BatchNormalization(
                epsilon=EPSILON, momentum=MOMENTUM
            ),
            tf.keras.layers.Activation(binary_tanh),
            BinaryConv2D(128, 3, H=H, padding="same", use_bias=False),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.BatchNormalization(
                epsilon=EPSILON, momentum=MOMENTUM
            ),
            tf.keras.layers.Activation(binary_tanh),
            BinaryConv2D(128, 3, H=H, padding="same", use_bias=False),
            tf.keras.layers.MaxPooling2D(2),
            tf.keras.layers.BatchNormalization(
                epsilon=EPSILON, momentum=MOMENTUM
            ),
            tf.keras.layers.Activation(binary_tanh),
            tf.keras.layers.Flatten(),
            BinaryDense(256, H=H, use_bias=False),
            tf.keras.layers.BatchNormalization(
                epsilon=EPSILON, momentum=MOMENTUM
            ),
            tf.keras.layers.Activation(binary_tanh),
            BinaryDense(NUM_CLASSES, H=H, use_bias=False),
            tf.keras.layers.BatchNormalization(
                epsilon=EPSILON, momentum=MOMENTUM
            ),
        ],
        name="mnist_cbcnn",
    )


def save_history_plot(history):
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="train")
    axes[0].plot(history.history["val_accuracy"], label="validation")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="train")
    axes[1].plot(history.history["val_loss"], label="validation")
    axes[1].set_title("Loss")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "training_history.png", dpi=150)
    plt.close(figure)


def train_model(epochs=50, batch_size=50):
    tf.keras.utils.set_random_seed(SEED)
    (x_train, y_train), (x_test, y_test) = load_data()
    model = build_model()

    lr_decay = (LR_END / LR_START) ** (1.0 / max(epochs, 1))
    checkpoint_path = OUTPUT_DIR / "mnist_cbcnn.best.weights.h5"
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch: LR_START * lr_decay**epoch
        ),
        tf.keras.callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor="val_accuracy",
            save_best_only=True,
            save_weights_only=True,
            mode="max",
        ),
    ]
    model.compile(
        loss="squared_hinge",
        optimizer=tf.keras.optimizers.Adam(learning_rate=LR_START),
        metrics=[tf.keras.metrics.CategoricalAccuracy(name="accuracy")],
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1,
    )
    model.load_weights(checkpoint_path)
    score = model.evaluate(x_test, y_test, verbose=1)
    model.save(OUTPUT_DIR / "mnist_cbcnn.keras")
    save_history_plot(history)
    print(f"Test loss: {score[0]:.6f}")
    print(f"Test accuracy: {score[1]:.6f}")
    return model, history, score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.epochs, args.batch_size)

