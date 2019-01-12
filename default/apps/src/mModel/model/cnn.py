# Simple CNN model for CIFAR-10
import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from default.apps.src.mModel.model.modelManager import ModelManager
K.set_image_dim_ordering('th')


class Cnn(ModelManager):

    def __init__(self, hyperParameter, nb_epoch, batch_size, nb_classes, dataset):
        ModelManager.__init__(self, hyperParameter, nb_epoch, batch_size, nb_classes, dataset)
        self.__hyperParameter = hyperParameter
        self.__nb_epoch = nb_epoch
        self.__batch_size = batch_size
        self.__dataset = dataset

    def run_model_sample_cnn(self):
        # load data
        (X_train, y_train), (X_test, y_test) = self.__dataset
        # normalize inputs from 0-255 to 0.0-1.0
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_train = X_train / 255.0
        X_test = X_test / 255.0

        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        nb_classes = y_test.shape[1]

        # Compile model
        lrate = 0.01
        decay = lrate / self.__nb_epoch

        # Create the model
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.2))
        model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
        model.add(Dropout(0.5))
        model.add(Dense(nb_classes, activation='softmax'))

        sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
        model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

        model.summary()

        # Fit the model
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs=self.__nb_epoch,
                  batch_size=self.__batch_size)

        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
