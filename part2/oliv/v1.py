import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import os
from tensorflow.python.keras.backend import set_floatx
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.utils import to_categorical
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

data = pd.read_csv('../train.csv')

data['category_id'] = data['category_id'].astype('str')
data = data.sample(frac=1)

# train_data = data.iloc[:137409, :]


# valid_data = data.iloc[137409:, :]


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


dropout_dense_layer = 0.6

model = Sequential()
model.add(Conv2D(32, (7, 7),
                 padding='same',
                 kernel_constraint=maxnorm(3),
                 kernel_regularizer=regularizers.l2(1e-4),
                 input_shape=(32, 32, 3)))

model.add(LeakyReLU(alpha=0.3))
#model.add(BatchNormalization())

model.add(Conv2D(32, (7, 7),
                 padding='same',
                 kernel_constraint=maxnorm(3),
                 kernel_regularizer=regularizers.l2(1e-4)
                 ))
model.add(LeakyReLU(alpha=0.3))
#model.add(BatchNormalization())

model.add(Conv2D(32, (7, 7),
                 padding='same',
                 kernel_constraint=maxnorm(3),
                 kernel_regularizer=regularizers.l2(1e-4)
                 ))
model.add(LeakyReLU(alpha=0.3))
#model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(64, (7, 7),
                 padding='same',
                 kernel_constraint=maxnorm(3),
                 kernel_regularizer=regularizers.l2(1e-4)
                 ))
model.add(LeakyReLU(alpha=0.3))
#model.add(BatchNormalization())

model.add(Conv2D(64, (7, 7),
                 padding='same',
                 kernel_constraint=maxnorm(3),
                 kernel_regularizer=regularizers.l2(1e-4)
                 ))
model.add(LeakyReLU(alpha=0.3))
#model.add(BatchNormalization())

model.add(Conv2D(64, (7, 7),
                 padding='same',
                 kernel_constraint=maxnorm(3),
                 kernel_regularizer=regularizers.l2(1e-4)
                 ))
model.add(LeakyReLU(alpha=0.3))
#model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.3))

model.add(Conv2D(128, (7, 7),
                 padding='same',
                 kernel_constraint=maxnorm(3),
                 kernel_regularizer=regularizers.l2(1e-4)
                 ))
model.add(LeakyReLU(alpha=0.3))
#model.add(BatchNormalization())

model.add(Conv2D(128, (7, 7),
                 padding='same',
                 kernel_constraint=maxnorm(3),
                 kernel_regularizer=regularizers.l2(1e-4)
                 ))
model.add(LeakyReLU(alpha=0.3))
#model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('sigmoid'))
model.add(Dropout(dropout_dense_layer))

model.add(Dense(256))
model.add(Activation('sigmoid'))
model.add(Dropout(dropout_dense_layer))

model.add(Dense(14))
model.add(Activation('sigmoid'))

model.compile(loss='categorical_crossentropy',
              # optimizer=Adam(0.0001),
              optimizer='rmsprop',
              metrics=['accuracy'])

model.summary()

datagen = ImageDataGenerator(
    rescale=1. / 255,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.3
)

t_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory="../train_images/",
    x_col="file_name",
    y_col="category_id",
    shuffle=True,
    target_size=(32, 32),
    batch_size=128,
    class_mode="categorical",
    subset="training")

# v_datagen = ImageDataGenerator(
#    rescale=1. / 255
# )

v_generator = datagen.flow_from_dataframe(
    dataframe=data,
    directory="../train_images/",
    x_col="file_name",
    y_col="category_id",
    target_size=(32, 32),
    batch_size=128,
    class_mode="categorical",
    subset="validation")

# Fit the model
# filepath="weights_resnet.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#learning_rate_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.6, patience=10, verbose=1, mode='max',
#                                         min_delta=0.0, cooldown=0, min_lr=0)
#early_stop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=80, verbose=1, mode='max', baseline=None,
#                           restore_best_weights=True)
callbacks = [EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', verbose=1, save_best_only=True),
             ReduceLROnPlateau(monitor='val_acc', factor=0.6, patience=3, verbose=1,
                               mode='auto', min_delta=0.0, cooldown=0, min_lr=0)

             ]

history = model.fit_generator(t_generator,
                              validation_data=v_generator,
                              epochs=400,
                              callbacks=callbacks,
                              shuffle=True,
                              verbose=1
                              )

model.load_weights("best_model.h5")

# test_files = os.listdir('test/test/')
# x_test = []
# for i in range(len(test_files)):
#     img = mpimg.imread('test/test/' + test_files[i])
#     x_test.append(img)
# x_test = np.asarray(x_test)
# x_test = x_test.astype('float32')
# x_test /= 255.
# predict = model.predict(x_test)
# predict = np.array(predict).reshape((-1, 1))
# predict = predict.reshape(-1).tolist()
# pred = [0 if value < 0.50 else 1 for value in predict]
# sub_file = pd.DataFrame(
#     data={'id': test_files, 'has_cactus': pred})
# sub_file.to_csv('sample_submission6.csv', index=False)
