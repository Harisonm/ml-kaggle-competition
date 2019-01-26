# Simple CNN model for CIFAR-10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras import backend as K
from keras.utils import np_utils
from default.apps.src.mModel.manager.ModelManager import ModelManager
from keras.callbacks import TensorBoard
K.set_image_dim_ordering('th')


class Cnn(ModelManager):

    def __init__(self, param, dataset):
        super().__init__(param, dataset)
        self.__param = self._random_param(param)
        self.__dataset = self._preprocess_cifar10(dataset)

    def run_model(self):
        # load data

        (X_train, y_train), (X_test, y_test) = self.__dataset
        nb_classes = y_test.shape[1]

        # Compile model
        decay = self.__param['lr'] / self.__param['epochs']

        # Create the model
        model = Sequential()
        model.add(Conv2D(32, (3, 3),
                         input_shape=self.__param['input_shape_cnn'],
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

        model.add(Dense(self.__param['units'],
                        activation=self.__param['activation'],
                        kernel_constraint=self.__param['kernel_constraint']))

        model.add(Dropout(self.__param['dropout']))

        model.add(Dense(nb_classes,
                        activation=self.__param['activation']))

        model.compile(loss=self.__param['losses'],
                      optimizer=self.__param['optimizer'],
                      metrics=['accuracy'])

        model.summary()

        type_model = "cnn"
        tb_callback = self.__save_tensorboard(model, type_model)

        # Fit the model
        model.fit(X_train, y_train,
                  validation_data=(X_test, y_test),
                  epochs=self.__param['epochs'],
                  batch_size=self.__param['batch_size'],
                  callbacks=[tb_callback])

        # Final evaluation of the model
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

    def __save_tensorboard(self, model, type_model):
        model_str = type_model + "_" + \
                    str(self.__param['activation']) + "_" + \
                    str(self.__param['losses']) + "_" + \
                    str(self.__param['optimizer'])

        model.save("./tensorboard/" + type_model + "/saved_models" + "_" + model_str, True, True)

        # Save tensorboard callback
        tb_callback = TensorBoard(log_dir="./tensorboard/" + type_model + "/logs/" + type_model + "_" +
                                          str(self.__param['activation']) + "_" +
                                          str(self.__param['losses']) + "_" +
                                          str(self.__param['optimizer']))
        return tb_callback

    def _preprocess_cifar10(self, dataset):

        (__X_train, __y_train), (__X_test, __y_test) = dataset
        # normalize inputs from 0-255 to 0.0-1.0
        __X_train = __X_train.astype('float32')
        __X_test = __X_test.astype('float32')
        __X_train = __X_train / 255.0
        __X_test = __X_test / 255.0

        # one hot encode outputs
        __y_train = np_utils.to_categorical(__y_train)
        __y_test = np_utils.to_categorical(__y_test)

        return (__X_train, __y_train), (__X_test, __y_test)
