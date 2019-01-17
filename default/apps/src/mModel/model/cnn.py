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

    def __init__(self, param, dataset):

        ModelManager.__init__(self, param, dataset)
        self.__dataset = self.preprocess_cifar10(self)
        self.__param = self.random_param(param)

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
        decay = self.__param['lr'] / self.__param['epochs']

        # Create the model
        model = Sequential()
        model.add(Conv2D(32, (3, 3),
                         input_shape=self.__param['input_shape'],
                         padding=self.__param['padding'],
                         activation=self.__param['activation'],
                         kernel_constraint=self.__param['kernel_constraint']))

        model.add(Dropout(self.__param['dropout']))

        model.add(Conv2D(32, (3, 3),
                         activation=self.__param['activation'],
                         padding=self.__param['padding'],
                         kernel_constraint=self.__param['kernel_constraint']))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(self.__param['"layerParam")[''denseMiddle'],
                        activation=self.__param['activation'],
                        kernel_constraint=self.__param['kernel_constraint']))

        model.add(Dropout(self.__param['dropout']))

        model.add(Dense(nb_classes,
                        activation=self.__param['activation']))

        model.compile(loss=self.__param['losses'],
                      optimizer=self.__param['optimizer'],
                      metrics=['accuracy'])

        model.summary()

        type_model = "cnn1"
        tb_callback = self.save_tensorboard(model, type_model)

        # Fit the model
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs=self.__param['epochs'],
                  batch_size=self.__param['batch_size'],
                  callbacks=[tb_callback])

        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
