"""Train and evaluate a binary CNN on the GTSRB traffic-sign dataset."""

import argparse
import pickle
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from binary_layers import BinaryConv2D, BinaryDense
from binary_ops import binary_tanh
from util import load_traffic_sign_data

SEED = 37
IMAGE_SHAPE = (32, 32, 1)
NUM_CLASSES = 43
H = 1.0
LR_START = 1e-3
LR_END = 1e-4
EPSILON = 1e-6
MOMENTUM = 0.9
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "traffic-signs-data"


def _to_equalized_luminance(images):
    return np.asarray(
        [
            cv2.equalizeHist(
                cv2.cvtColor(image, cv2.COLOR_RGB2YUV)[:, :, 0]
            )[..., np.newaxis]
            for image in images
        ],
        dtype=np.float32,
    )


def get_mean_std_img(images):
    luminance = _to_equalized_luminance(images)
    mean_img = np.mean(luminance, axis=0)
    std_img = np.std(luminance, axis=0)
    std_img = np.where(std_img < 1e-6, 1.0, std_img)
    return mean_img.astype(np.float32), std_img.astype(np.float32)


def preprocess_features(images, mean_img, std_img):
    luminance = _to_equalized_luminance(images)
    return (luminance - mean_img) / std_img


def stratified_split(images, labels, validation_fraction=0.1):
    rng = np.random.default_rng(SEED)
    validation_indices = []
    for class_id in range(NUM_CLASSES):
        indices = np.flatnonzero(labels == class_id)
        if not len(indices):
            raise ValueError(f"Training data has no examples for class {class_id}.")
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * validation_fraction)))
        validation_indices.extend(indices[:count])

    validation_indices = np.asarray(validation_indices, dtype=np.int64)
    training_mask = np.ones(len(labels), dtype=bool)
    training_mask[validation_indices] = False
    return (
        images[training_mask],
        labels[training_mask],
        images[validation_indices],
        labels[validation_indices],
    )


def get_image_generator():
    return tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=15.0,
        zoom_range=0.2,
        width_shift_range=0.1,
        height_shift_range=0.1,
    )


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
        name="gtsrb_cbcnn",
    )


def save_history(history):
    with (BASE_DIR / "train_history.pkl").open("wb") as file:
        pickle.dump(history.history, file)

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
    figure.savefig(BASE_DIR / "training_history.png", dpi=150)
    plt.close(figure)


def train_model(epochs=1000, batch_size=200):
    tf.keras.utils.set_random_seed(SEED)
    x_all, y_all = load_traffic_sign_data(DATA_DIR / "train.p")
    x_test, y_test = load_traffic_sign_data(DATA_DIR / "test.p")

    x_train, y_train, x_validation, y_validation = stratified_split(
        x_all, y_all
    )
    mean_img, std_img = get_mean_std_img(x_train)
    x_train = preprocess_features(x_train, mean_img, std_img)
    x_validation = preprocess_features(x_validation, mean_img, std_img)
    x_test = preprocess_features(x_test, mean_img, std_img)

    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES) * 2.0 - 1.0
    y_validation = (
        tf.keras.utils.to_categorical(y_validation, NUM_CLASSES) * 2.0 - 1.0
    )
    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES) * 2.0 - 1.0

    model = build_model()
    lr_decay = (LR_END / LR_START) ** (1.0 / max(epochs, 1))
    checkpoint_path = BASE_DIR / "gtsrb_cbcnn.best.weights.h5"
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
    training_flow = get_image_generator().flow(
        x_train, y_train, batch_size=batch_size, shuffle=True, seed=SEED
    )
    history = model.fit(
        training_flow,
        epochs=epochs,
        validation_data=(x_validation, y_validation),
        callbacks=callbacks,
        verbose=1,
    )
    model.load_weights(checkpoint_path)
    score = model.evaluate(x_test, y_test, verbose=1)
    model.save(BASE_DIR / "gtsrb_cbcnn.keras")
    save_history(history)
    print(f"Test loss: {score[0]:.6f}")
    print(f"Test accuracy: {score[1]:.6f}")
    return model, history, score


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=200)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_model(args.epochs, args.batch_size)

