from default.apps.src.mModel.model.modelManager import ModelManager
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.optimizers import SGD


class Mlp(ModelManager):

    def __init__(self, param, nb_epoch, batch_size, nb_classes, dataset):
        '''
        :param param:
        :param nb_epoch:
        :param batch_size:
        :param nb_classes:
        :param dataset:
        '''
        ModelManager.__init__(self, param, nb_epoch, batch_size, nb_classes, dataset)
        self.__param = param
        self.__nb_epoch = nb_epoch
        self.__batch_size = batch_size
        self.__nb_classes = nb_classes
        self.__dataset = ModelManager.preprocess(self)

    def run_model(self):
        '''
        :return:
        '''
        (X_train, y_train), (X_test, y_test) = self.__dataset

        # MLP
        model = Sequential()
        model.add(Dense(1024,
                        input_shape=(self.__param['input_shape'],),
                        activation=self.__param['activation'],
                        kernel_constraint=self.__param['kernel_constraint']))

        model.add(Dropout(self.__param['dropout']))

        ## Hidding layer
        model.add(Dense(512,
                        activation=self.__param['activation'],
                        kernel_constraint=self.__param['kernel_constraint']))

        model.add(Dropout(self.__param['dropout']))

        model.add(Dense(512,
                        activation=self.__param['activation'],
                        kernel_constraint=self.__param['kernel_constraint']))

        model.add(Dropout(self.__param['dropout']))

        ## End hidden layer
        model.add(Dense(10,
                        activation=self.__param['last_activation']))

        # Compile model
        model.compile(loss=self.__param['losses'],
                      optimizer=self.__param['optimizer'],
                      metrics=self.__param['metrics'])

        model.summary()

        type_model = "mlp1"
        tb_callback = self.save_tensorboard(model, type_model)

        # training
        history = model.fit(X_train, y_train,
                            batch_size=self.__batch_size,
                            epochs=self.__nb_epoch,
                            verbose=1,
                            validation_data=(X_test, y_test),
                            callbacks=[tb_callback])

        self.save_history(history, 'history.txt')

        loss, acc = model.evaluate(X_test, y_test, verbose=1)
        print('Test loss:', loss)
        print('Test acc:', acc)
