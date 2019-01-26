from default.apps.src.mModel.manager.ModelManager import ModelManager
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import TensorBoard


class Slp(ModelManager):

    def __init__(self, param, dataset):
        super().__init__(param, dataset)
        self.__param = self._random_param(param)
        self.__dataset = self._preprocess_cifar10(dataset)

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

        tb_callback = self.__save_tensorboard(model, type_model)
        # training
        history = model.fit(X_train, y_train,
                            batch_size=self.__param['batch_size'],
                            epochs=self.__param['epochs'],
                            verbose=1,
                            validation_data=(X_test, y_test),
                            callbacks=[tb_callback])

        self._save_history(history, "history/" + type_model + "/" + str(self.__param['activation']) + "_" +
                           str(self.__param['losses']) + str(self.__param['metrics']) + "_" + 'history.txt')

        loss, acc = model.evaluate(X_test, y_test, verbose=1)
        print('test loss:', loss)
        print('test acc:', acc)

        return history, model

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
