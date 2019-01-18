from default.apps.src.mModel.model.modelManager import ModelManager
from keras.layers import Dense, Dropout
from keras.models import Sequential
from numpy.random import random, randint
from keras.optimizers import SGD
import random


class Mlp(ModelManager):

    def __init__(self, param, dataset):

        ModelManager.__init__(self, param, dataset)
        self.__dataset = self.preprocess_cifar10(self)
        self.__param = self.random_param(param)

    def run_model(self):

        (X_train, y_train), (X_test, y_test) = self.__dataset
        type_model = "mlp"

        # MLP
        model = Sequential()

        # training Model
        model.add(Dense(1024,
                        input_shape=(self.__param['input_shape'],),
                        activation=self.__param['activation'],
                        kernel_constraint=self.__param['kernel_constraint']))

        model.add(Dropout(self.__param['dropout']))

        for layer in range(self.__param['hidden_layers']):
            model.add(Dense(512,
                            activation=self.__param['activation'],
                            kernel_constraint=self.__param['kernel_constraint']))
            model.add(Dropout(self.__param['dropout']))

        # End hidden layer
        model.add(Dense(10,
                        activation=self.__param['last_activation']))

        # Compile model
        model.compile(loss=self.__param['losses'],
                      optimizer=self.__param['optimizer'],
                      metrics=self.__param['metrics'])

        model.summary()
        tb_callback = self.save_tensorboard(model, type_model)

        # training
        history = model.fit(X_train, y_train,
                            batch_size=self.__param['batch_size'],
                            epochs=self.__param['epochs'],
                            verbose=1,
                            validation_data=(X_test, y_test),
                            callbacks=[tb_callback])

        self.save_history(history,
                          "history/" + type_model + "/" + self.__param['activation'] + "_" + self.__param['losses'] +
                          self.__param['metrics'] + "_" + '_history.txt')

        loss, acc = model.evaluate(X_test, y_test, verbose=1)
        print('test loss:', loss)
        print('test acc:', acc)
