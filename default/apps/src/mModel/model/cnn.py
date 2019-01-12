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
        '''
        :param hyperParameter:
        :param nb_epoch:
        :param batch_size:
        :param nb_classes:
        :param dataset:
        '''
        ModelManager.__init__(self, hyperParameter, nb_epoch, batch_size, nb_classes, dataset)
        self.__hyperParameter = hyperParameter
        self.__nb_epoch = nb_epoch
        self.__batch_size = batch_size
        self.__dataset = dataset

    def run_model_sample_cnn(self):
        '''
        :return:
        '''
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
        model.add(Conv2D(32, (3, 3),
                         input_shape=self.__hyperParameter.get("input_shape"),
                         padding=self.__hyperParameter.get("padding"),
                         activation=self.__hyperParameter.get("activation_1"),
                         kernel_constraint=maxnorm(3)))

        model.add(Dropout(self.__hyperParameter.get("dropout").get("param1")))

        model.add(Conv2D(32, (3, 3),
                         activation=self.__hyperParameter.get("activation_1"),
                         padding=self.__hyperParameter.get("padding"),
                         kernel_constraint=maxnorm(3)))

        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())

        model.add(Dense(self.__hyperParameter.get("layerParam").get("denseMiddle"),
                        activation=self.__hyperParameter.get("activation_1"),
                        kernel_constraint=maxnorm(3)))

        model.add(Dropout(self.__hyperParameter.get("dropout").get("param2")))

        model.add(Dense(nb_classes,
                        activation=self.__hyperParameter.get("activation_2")))

        sgd = SGD(lr=lrate,
                  momentum=0.9,
                  decay=decay,
                  nesterov=False)

        model.compile(loss=self.__hyperParameter.get("loss"),
                      optimizer=sgd,
                      metrics=['accuracy'])

        model.summary()

        type_model = "cnn1"
        tb_callback = super().save_tensorboard(model, type_model)

        # Fit the model
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs=self.__nb_epoch,
                  batch_size=self.__batch_size,
                  callbacks=[tb_callback])

        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1]*100))
