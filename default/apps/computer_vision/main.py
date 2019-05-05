import os
import json
import pandas as pd
import numpy as np
import os
import json
import logging
import datetime
import warnings
import seaborn as sns
import cv2
import random
from PIL import Image
import tensorflow as tf
import math
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras import layers
from tensorflow.python.keras.applications import DenseNet121
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import Callback, ModelCheckpoint
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import os

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

from glob import glob


def display_samples(df, columns=4, rows=3):
    fig = plt.figure(figsize=(5 * columns, 3 * rows))

    for i in range(columns * rows):
        image_path = df.loc[i, 'file_name']
        image_id = df.loc[i, 'category_id']
        img = cv2.imread(f'dataset/train_images/{image_path}')
        fig.add_subplot(rows, columns, i + 1)
        plt.title(image_id)
        plt.imshow(img)
        plt.show()


def get_pad_width(im, new_shape, is_rgb=True):
    pad_diff = new_shape - im.shape[0], new_shape - im.shape[1]
    t, b = math.floor(pad_diff[0] / 2), math.ceil(pad_diff[0] / 2)
    l, r = math.floor(pad_diff[1] / 2), math.ceil(pad_diff[1] / 2)
    if is_rgb:
        pad_width = ((t, b), (l, r), (0, 0))
    else:
        pad_width = ((t, b), (l, r))
    return pad_width


def pad_and_resize(image_path, dataset, pad=False, desired_size=32):
    img = cv2.imread(f'dataset/{dataset}_images/{image_path}.jpg')

    if pad:
        pad_width = get_pad_width(img, max(img.shape))
        padded = np.pad(img, pad_width=pad_width, mode='constant', constant_values=0)
    else:
        padded = img

    resized = cv2.resize(padded, (desired_size,) * 2).astype('uint8')

    return resized


def reduction_size():
    label_df = pd.read_csv('dataset/train.csv')
    submission_df = pd.read_csv('dataset/sample_submission.csv')
    print(label_df.head())
    label_df['category_id'].value_counts()[1:16].plot(kind='bar')

    display_samples(label_df)
    train_resized_imgs = []
    test_resized_imgs = []

    for image_id in label_df['id']:
        train_resized_imgs.append(
            pad_and_resize(image_id, 'train')
        )

    for image_id in submission_df['Id']:
        test_resized_imgs.append(
            pad_and_resize(image_id, 'test')
        )
    X_train = np.stack(train_resized_imgs)
    X_test = np.stack(test_resized_imgs)

    target_dummies = pd.get_dummies(label_df['category_id'])
    train_label = target_dummies.columns.values
    y_train = target_dummies.values

    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)

    # No need to save the IDs of X_test, since they are in the same order as the
    # ID column in sample_submission.csv
    np.save('dataset/reducing_image_sizes_to_32_32/X_train.npy', X_train)
    np.save('dataset/reducing_image_sizes_to_32_32/X_test.npy', X_test)
    np.save('dataset/reducing_image_sizes_to_32_32y_train.npy', y_train)


class Metrics(Callback):

    def __init__(self):
        super().__init__()

    def on_train_begin(self, logs={}):
        self.val_f1s = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        X_val, y_val = self.validation_data[:2]
        y_pred = self.model.predict(X_val)

        y_pred_cat = tf.python.keras.utils.to_categorical(
            y_pred.argmax(axis=1),
            num_classes=num_classes
        )

        _val_f1 = f1_score(y_val, y_pred_cat, average='macro')
        _val_recall = recall_score(y_val, y_pred_cat, average='macro')
        _val_precision = precision_score(y_val, y_pred_cat, average='macro')

        self.val_f1s.append(_val_f1)
        self.val_recalls.append(_val_recall)
        self.val_precisions.append(_val_precision)

        print((f"val_f1: {_val_f1:.4f}"
               f" — val_precision: {_val_precision:.4f}"
               f" — val_recall: {_val_recall:.4f}"))

        return


if __name__ == '__main__':
    # reduction_size()
    x_train = np.load('dataset/reducing_image_sizes_to_32_32/X_train.npy')
    x_test = np.load('dataset/reducing_image_sizes_to_32_32/X_test.npy')
    y_train = np.load('dataset/reducing_image_sizes_to_32_32/y_train.npy')

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    # Convert the images to float and scale it to a range of 0 to 1
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255.
    x_test /= 255.

    # A MODIFIER
    batch_size = 64
    num_classes = 14
    epochs = 30
    val_split = 0.1
    save_dir = os.path.join(os.getcwd(), 'models')
    model_name = 'keras_cnn_model.h5'

    # A SUPPRIMER
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    f1_metrics = Metrics()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )

    hist = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[f1_metrics],
        validation_split=val_split,
        shuffle=True
    )

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_path = os.path.join(save_dir, model_name)
    model.save(model_path)
    print('Saved trained model at %s ' % model_path)

    # 4. Evaluation
    history_df = pd.DataFrame(hist.history)
    history_df['val_f1'] = f1_metrics.val_f1s
    history_df['val_precision'] = f1_metrics.val_precisions
    history_df['val_recall'] = f1_metrics.val_recalls

    history_df[['loss', 'val_loss']].plot()
    history_df[['acc', 'val_acc']].plot()
    history_df[['val_f1', 'val_precision', 'val_recall']].plot()
    plt.show()

    # 5. Submission
    y_test = model.predict(x_test)

    submission_df = pd.read_csv('dataset/sample_submission.csv')
    submission_df['Predicted'] = y_test.argmax(axis=1)
    print(submission_df.shape)
    submission_df.head()

    submission_df.to_csv('submission.csv', index=False)
    history_df.to_csv('history.csv', index=False)

    with open('history.json', 'w') as f:
        json.dump(hist.history, f)
