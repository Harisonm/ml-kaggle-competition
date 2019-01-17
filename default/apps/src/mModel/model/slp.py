from default.apps.src.mModel.model.modelManager import ModelManager
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import TensorBoard


class Slp(ModelManager):

    def __init__(self, param, dataset):

        ModelManager.__init__(self, param, dataset)
        self.__dataset = self.preprocess_cifar10(self)
        self.__param = self.random_param(param)

    def run_model(self):
        '''
        :return:
        '''
        (X_train, y_train), (X_test, y_test) = self.__dataset
        type_model = "slp"

        model = Sequential()
        model.add(Dense(self.__param['unitsSlp'],
                        input_shape=(self.__param['input_shape'],),
                        activation=self.__param['activation']))

        model.compile(loss=self.__param['losses'],
                      optimizer=self.__param['optimizer'],
                      metrics=self.__param['metrics'])
        model.summary()

        tb_callback = super().save_tensorboard(model, type_model)
        # training
        history = model.fit(X_train, y_train,
                            batch_size=self.__param['batch_size'],
                            epochs=self.__param['epochs'],
                            verbose=1,
                            validation_data=(X_test, y_test),
                            callbacks=[tb_callback])

        self.save_history(history, "history/" + type_model + "/" + str(self.__param['activation']) + "_" +
                          str(self.__param['losses']) + str(self.__param['metrics']) + "_" + 'history.txt')

        loss, acc = model.evaluate(X_test, y_test, verbose=1)
        print('Test loss:', loss)
        print('Test acc:', acc)

        return history, model

