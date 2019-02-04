from default.apps.src.mModel.manager.ModelManager import ModelManager
from keras.layers import Dense
from keras.models import Sequential
from keras.callbacks import TensorBoard
import os
import sys
import mlflow
import mlflow.keras

PATH_TB = "tensorboard/"
PATH_HISTORY = "history/"


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

        self._save_history(history, PATH_HISTORY + type_model + "/" + str(self.__param['activation']) + "_" +
                           str(self.__param['losses']) + str(self.__param['metrics']) + "_" + 'history.txt')

        score = model.evaluate(X_test, y_test, verbose=1)
        print('test loss:', score[0])
        print('test acc:', score[1])

        self.run_mlflow(self.__param, history, model, score)
        return history, model

    def __save_tensorboard(self, model, type_model):
        model_str = type_model + "_" + \
                    str(self.__param['epochs']) + "_" + \
                    str(self.__param['batch_size']) + "_" + \
                    str(self.__param['activation']) + "_" + \
                    str(self.__param['losses']) + "_" + \
                    str(self.__param['optimizer'])

        model.save(PATH_TB + type_model + "/saved_models" + "_" + model_str, True, True)

        # Save tensorboard callback
        tb_callback = TensorBoard(log_dir="./tensorboard/" + type_model + "/logs/" + type_model + "_" +
                                          str(self.__param['epochs']) + "_" +
                                          str(self.__param['batch_size']) + "_" +
                                          str(self.__param['activation']) + "_" +
                                          str(self.__param['losses']) + "_" +
                                          str(self.__param['optimizer']))
        return tb_callback

    def run_mlflow(self, param, history, model, score):
        with mlflow.start_run():
            # log parameters
            mlflow.log_param("hidden_layers", self.__param['unitsSlp'])
            mlflow.log_param("input_shape", self.__param['input_shape'])
            mlflow.log_param("activation", self.__param['activation'])
            mlflow.log_param("epochs", self.__param['epochs'])
            mlflow.log_param("loss_function", self.__param['losses'])

            # calculate metrics
            binary_loss = self._get_binary_loss(history)
            binary_acc = self._get_binary_acc(history, param)
            validation_loss = self._get_validation_loss(history)
            validation_acc = self._get_validation_acc(history)
            average_loss = score[0]
            average_acc = score[1]

            # log metrics
            mlflow.log_metric("binary_loss", binary_loss)
            mlflow.log_metric(self.__param['metrics'], binary_acc)
            mlflow.log_metric("validation_loss", validation_loss)
            mlflow.log_metric("validation_acc", validation_acc)
            mlflow.log_metric("average_loss", average_loss)
            mlflow.log_metric("average_acc", average_acc)

            # log artifacts (matplotlib images for loss/accuracy)
            # mlflow.log_artifacts(image_dir)
            # log model
            mlflow.keras.log_model(model, "models")

        print("loss function use", self.__param['losses'])
        pass
