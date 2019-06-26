import sys
from numpy.random import random
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import LeakyReLU
from mlflow_builder import MLFlowBuilder
import pandas as pd
import tensorflow as tf
import numpy as np
import time
from tensorflow.python.keras.callbacks import TensorBoard
from mlflow_builder.MLFlowBuilder import MLFlowBuilder
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
from tensorflow.python.keras.optimizers import Adam, RMSprop
import tensorflow as tf
from tensorflow.python.keras.constraints import maxnorm
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.python.keras.optimizers import SGD
from mlflow_builder.MLFlowBuilder import MLFlowBuilder

PATH = 'dataset/cactus'


def run_model(param):
    train_df = pd.read_csv(PATH, 'train.csv')
    epochs = param.get('epochs')

    print(train_df.head())
    train_df['has_cactus'] = train_df['has_cactus'].astype(str)
    train_df = train_df.sample(frac=1)

    train_data = train_df.iloc[:15750, :]
    valid_data = train_df.iloc[15750:, :]

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        horizontal_flip=True,
        vertical_flip=True,
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_data,
        directory="dataset/cactus/train/",
        x_col="id",
        y_col="has_cactus",
        shuffle=True,
        target_size=(32, 32),
        batch_size=64,
        class_mode='binary')

    v_datagen = ImageDataGenerator(
        rescale=1. / 255
    )

    valid_generator = v_datagen.flow_from_dataframe(
        dataframe=valid_data,
        directory="dataset/cactus/train/",
        x_col="id",
        y_col="has_cactus",
        target_size=(32, 32),
        batch_size=64,
        class_mode='binary')

    dropout_dense_layer = 0.6

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation(param.get('relu')))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(param.get('relu')))
    model.add(Conv2D(32, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(param.get('relu')))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(param.get('relu')))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(param.get('relu')))
    model.add(Conv2D(64, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(param.get('relu')))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation(param.get('relu')))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation(param.get('relu')))
    model.add(Dropout(dropout_dense_layer))

    model.add(Dense(256))
    model.add(Activation(param.get('relu')))
    model.add(Dropout(dropout_dense_layer))

    model.add(Dense(1))
    model.add(Activation(param.get('last_activation')))

    model.compile(loss=param.get('losses'),
                  optimizer=param.get('optimizer'),
                  # optimizer='rmsprop',
                  metrics=param.get('metrics'))

    model.summary()
    type_model = "CNN_model_1"

    # tb_callback = self.__save_tensorboard(model, type_model)
    tb_callback = TensorBoard(log_dir='logs/' + type_model + (str(round(time.time()))))

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  epochs=param.get('epochs'),
                                  workers=4,
                                  verbose=1,
                                  callbacks=[tb_callback]
                                  )

    model.save('model_save/my_model.h5')
    model.save_weights('model_save/my_model_weights.h5')

    history_df = pd.DataFrame(history.history)
    # history_df[['loss', 'val_loss']].plot()
    # history_df[['acc', 'val_acc']].plot()

    # Submission
    submission_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))

    # Evaluation
    # with open('history.json', 'w') as f:
    #     json.dump(history.history, f)
    # Final evaluation of the model
    score = model.evaluate_generator(valid_generator,
                                     steps=None,
                                     max_queue_size=10,
                                     workers=1,
                                     use_multiprocessing=False,
                                     verbose=0)

    MLFlowBuilder().run_ml_flow(type_model, param, history, model, score)


if __name__ == '__main__':
    epochs = 1
    # int(sys.argv[1])

    lr = random() * (0.1 - 0.0001) + 0.0001
    momentum = random() * (0.1 - 0.0001) + 0.0001
    decay = lr / epochs

    param = {'lr': lr,
             'optimizer': SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False),
             'losses': 'binary_crossentropy',
             'last_activation': 'softmax',
             'activation': 'relu',
             'metrics': ['accuracy'],
             'epochs': epochs,
             }

    run_model(param)
