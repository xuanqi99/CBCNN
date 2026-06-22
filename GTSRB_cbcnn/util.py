import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(23)

def load_traffic_sign_data(training_file):
    with open(training_file, mode='rb') as f:
        train = pickle.load(f)
    X_train, y_train = train['features'], train['labels']

    return X_train, y_train


def load_traffic_Tsign_data(training_file):
    with open(training_file, mode='rb') as T:
        test = pickle.load(T)
    X_test, y_test = test['features'], test['labels']

    return X_test, y_test


def show_random_samples(X_train, y_train, n_classes):
    # show a random sample from each class of the traffic sign dataset
    rows, cols = 4, 12
    fig, ax_array = plt.subplots(rows, cols)
    plt.suptitle('Train Samples (one per class)')
    for class_idx, ax in enumerate(ax_array.ravel()):
        if class_idx < n_classes:
            # show a random image of the current class
            cur_X = X_train[y_train == class_idx]
            cur_img = cur_X[np.random.randint(len(cur_X))]
            ax.imshow(cur_img)
            ax.set_title('{:02d}'.format(class_idx))
        else:
            ax.axis('off')
    # hide both x and y ticks
    plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
    plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
    plt.draw()


def show_test_samples(X_test, y_test, m_classes):
    # show a random sample from each class of the traffic sign dataset
    rows, cols = 4, 12
    fig, ax_array = plt.subplots(rows, cols)
    plt.suptitle('Test Samples (one per class)')
    for class_idx, ax in enumerate(ax_array.ravel()):
        if class_idx < m_classes:
            # show a random image of the current class
            cur_T = X_test[y_test == class_idx]
            cur_Timg = cur_T[np.random.randint(len(cur_T))]
            ax.imshow(cur_Timg)
            ax.set_title('{:02d}'.format(class_idx))
        else:
            ax.axis('off')
    # hide both x and y ticks
    plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
    plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
    plt.draw()


def show_classes_distribution(n_classes, y_train, n_train):
    # bar-chart of classes distribution
    train_distribution = np.zeros(n_classes)
    for c in range(n_classes):
        train_distribution[c] = np.sum(y_train == c) / n_train
    fig, ax = plt.subplots()
    col_width = 1
    bar_train = ax.bar(np.arange(n_classes), train_distribution, width=col_width, color='r')
    ax.set_ylabel('Train Percentage')
    ax.set_xlabel('Train Class Label')
    ax.set_title('Train Distribution')
    ax.set_xticks(np.arange(0, n_classes, 5) + col_width)
    ax.set_xticklabels(['{:02d}'.format(c) for c in range(0, n_classes, 5)])
    plt.show()


def show_Tclasses_distribution(m_classes, y_test, n_test):
    # bar-chart of classes distribution
    test_distribution = np.zeros(m_classes)
    for c in range(m_classes):
        test_distribution[c] = np.sum(y_test == c) / n_test
    fig, ax = plt.subplots()
    col_width = 1
    bar_test = ax.bar(np.arange(m_classes), test_distribution, width=col_width, color='g')
    ax.set_ylabel('Test Percentage')
    ax.set_xlabel('Test Class Label')
    ax.set_title('Test Distribution')
    ax.set_xticks(np.arange(0, m_classes, 5) + col_width)
    ax.set_xticklabels(['{:02d}'.format(c) for c in range(0, m_classes, 5)])
    plt.show()

def show_samples_from_generator(image_datagen, X_train, y_train):
    # take a random image from the training set
    img_rgb = X_train[0]
    # plot the original image
    plt.figure(figsize=(1, 1))
    plt.imshow(img_rgb)
    plt.title('Example of RGB image (class = {})'.format(y_train[0]))
    plt.show()
    # plot some randomly augmented images
    rows, cols = 4, 10
    fig, ax_array = plt.subplots(rows, cols)
    for ax in ax_array.ravel():
        augmented_img, _ = next(
            image_datagen.flow(np.expand_dims(img_rgb, 0), y_train[0:1])
        )
        ax.imshow(np.uint8(np.squeeze(augmented_img)))
    plt.setp([a.get_xticklabels() for a in ax_array.ravel()], visible=False)
    plt.setp([a.get_yticklabels() for a in ax_array.ravel()], visible=False)
    plt.suptitle('Random examples of data augmentation (starting from the previous image)')
    plt.show()


def get_image_generator():
    # create the generator to perform online data augmentation
    image_datagen = ImageDataGenerator(rotation_range=15.,
                                       zoom_range=0.2,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1)

    return image_datagen


if __name__ == "__main__":
    X_train, y_train = load_traffic_sign_data('./traffic-signs-data/train.p')
    X_test, y_test = load_traffic_Tsign_data('./traffic-signs-data/test.p')

    # Number of examples
    n_train = X_train.shape[0]
    n_test = X_test.shape[0]

    # What's the shape of an traffic sign image?
    image_shape = X_train[0].shape
    Timage_shape = X_test[0].shape

    # How many classes?
    n_classes = np.unique(y_train).shape[0]
    m_classes = np.unique(y_test).shape[0]


    image_datagen = ImageDataGenerator(rotation_range=15.,
                                       zoom_range=0.2,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1)

    print("训练数据集的数据个数 =", n_train)
    print("训练图像尺寸  =", image_shape)
    print("训练类别数量 =", n_classes)

    print("测试数据集的数据个数 =", n_test)
    print("测试图像尺寸  =", Timage_shape)
    print("测试类别数量 =", m_classes)

    show_random_samples(X_train, y_train, n_classes)
    show_test_samples(X_test, y_test, m_classes)
    show_classes_distribution(n_classes, y_train, n_train)
    show_Tclasses_distribution(m_classes, y_test, n_test)
    show_samples_from_generator(image_datagen, X_train, y_train)
