import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import LeakyReLU
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
from tensorflow.python.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.python.keras.constraints import maxnorm
from mlflow_builder.MLFlowBuilder import MLFlowBuilder
from numpy.random import random
from tensorflow.python.keras.optimizers import SGD
import sys

PATH_TRAIN = 'dataset/cactus/train.csv'
PATH = 'dataset/cactus/train/'


def run_model(param):
    type_model = "model_1_CNN"
    train_data = pd.read_csv(PATH_TRAIN)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    # print(train_data.shape)
    # vérifier fichier

    # si rien d'autre que 0 ou 1
    # print(train_data.has_cactus.unique())

    # Taille des images

    length = len(train_data)
    # print(length)
    x_train = []

    for i in range(length):
        img = mpimg.imread(PATH + train_data.id[i])
        x_train.append(img)
    # print(np.reshape(img, 32*32*3).shape)
    # img.shape
    # x,y,_ = img.shape
    # if x != 32 and y != 32:
    #	print(error)
    # print('END')

    # création des array

    # x_train
    y_train = train_data.has_cactus
    y_train = np.asarray(y_train)
    x_train = np.asarray(x_train)
    x_train = x_train.astype('float32')
    x_train /= 255.

    model = Sequential()

    model.add(Conv2D(32, (5, 5),
                     padding='same',
                     kernel_constraint=maxnorm(3),
                     # kernel_regularizer=regularizers.l2(1e-4),
                     input_shape=(32, 32, 3)))
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())

    model.add(Conv2D(32, (5, 5),
                     padding='same',
                     kernel_constraint=maxnorm(3),
                     # kernel_regularizer=regularizers.l2(1e-4)
                     ))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (5, 5),
                     padding='same',
                     kernel_constraint=maxnorm(3),
                     # kernel_regularizer=regularizers.l2(1e-4)
                     )
              )
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())

    model.add(Conv2D(64, (5, 5),
                     padding='same',
                     kernel_constraint=maxnorm(3),
                     # kernel_regularizer=regularizers.l2(1e-4)
                     )
              )
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.3))

    model.add(Conv2D(128, (5, 5),
                     padding='same',
                     kernel_constraint=maxnorm(3),
                     # kernel_regularizer=regularizers.l2(1e-4)
                     ))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())

    model.add(Conv2D(128, (5, 5),
                     padding='same',
                     kernel_constraint=maxnorm(3),
                     # kernel_regularizer=regularizers.l2(1e-4)
                     ))
    model.add(LeakyReLU(alpha=0.3))
    model.add(BatchNormalization())

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.4))

    model.add(Flatten())

    model.add(Dense(512,
                    activation='sigmoid',
                    kernel_constraint=maxnorm(2)))

    model.add(Dense(1,
                    activation='sigmoid'))

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(0.0001),
                  metrics=['accuracy'])

    model.summary()

    # Fit the model
    history = model.fit(x_train, y_train,
                        # steps_per_epoch=750,
                        validation_split=0.3,
                        epochs=1,
                        batch_size=128)

    # Final evaluation of the model
    score = model.evaluate(X_test, y_test, verbose=1)
    MLFlowBuilder().run_ml_flow(type_model, param, history, model)

    test_files = os.listdir('dataset/cactus/test/')
    x_test = []
    for i in range(len(test_files)):
        img = mpimg.imread('dataset/cactus/test/' + test_files[i])
        x_test.append(img)
    predict = model.predict(np.array(x_test))
    print(predict.shape)
    predict = np.array(predict).reshape((-1, 1))
    sub_file = pd.DataFrame(data={'id': test_files, 'has_cactus': predict.reshape(-1).tolist()})
    sub_file.to_csv('result_submission/sample_submission.csv', index=False)


if __name__ == '__main__':
    epochs = int(sys.argv[1])

    lr = random() * (0.1 - 0.0001) + 0.0001
    momentum = random() * (0.1 - 0.0001) + 0.0001
    decay = lr / epochs

    param = {'lr': lr,
             'optimizer': SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False),
             'losses': 'categorical_crossentropy',
             'last_activation': 'softmax',
             'metrics': 'acc',
             'epochs': epochs,
             }

    run_model(param)
