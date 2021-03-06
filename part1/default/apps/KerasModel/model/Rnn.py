from part1.default.apps.KerasModel.manager import ModelManager
from mlflow_builder import MLFlowBuilder
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.utils import np_utils
import numpy

PATH_TB = "./logsModel/tensorboard/"
PATH_HISTORY = "./logsModel/history/"


class Rnn(ModelManager, MLFlowBuilder):

    def __init__(self, param, dataset):
        """
        :param param:
        :param dataset:
        """
        super().__init__(param, dataset)
        self.__param = self._random_param(param)
        self.__dataset = self._preprocess_cifar10(dataset)

    def run_model(self):
        '''
        :return:
        '''
        (X_train, y_train), (X_test, y_test) = self.__dataset

        nb_classes = y_test.shape[1]
        type_model = "rnn"

        # Create the model
        # A MODIFIER : Mettre une condition pour construire des modèles sans Séquentiel et d'autre avec Sequential
        model = Sequential()

        model.add(LSTM(32,
                       input_shape=self.__param['input_shape_rnn'],
                       activation=self.__param['activation'],
                       kernel_constraint=self.__param['kernel_constraint'],
                       return_sequences=True
                       ))

        model.add(Dropout(self.__param['dropout']))

        model.add(LSTM(32,
                       activation=self.__param['activation'],
                       kernel_constraint=self.__param['kernel_constraint']))

        model.add(Dropout(self.__param['dropout']))

        model.add(Dense(nb_classes,
                        activation=self.__param['last_activation']))

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
                            callbacks=[tb_callback]
                            )

        # Final evaluation of the model
        score = model.evaluate(X_test, y_test, verbose=1)

        print('test loss:', score[0])
        print('test acc:', score[1])

        self._run_ml_flow(type_model, self.__param, history, model, score)
        return history, model

    def __save_tensorboard(self, model, type_model):
        """
        :param model:
        :param type_model:
        :return:
        """
        model_str = type_model + "_" + \
                    str(self.__param['epochs']) + "_" + \
                    str(self.__param['batch_size']) + "_" + \
                    str(self.__param['activation']) + "_" + \
                    str(self.__param['losses']) + "_" + \
                    str(self.__param['optimizer'])

        model.save(PATH_TB + type_model + "/saved_models" + "_" + model_str, True, True)

        # Save tensorboard callback
        tb_callback = TensorBoard(log_dir=str(PATH_TB) + "/" + type_model + "/" + type_model + "_" +
                                          str(self.__param['epochs']) + "_" +
                                          str(self.__param['batch_size']) + "_" +
                                          str(self.__param['activation']) + "_" +
                                          str(self.__param['losses']) + "_" +
                                          str(self.__param['optimizer']))
        return tb_callback

    def _preprocess_cifar10(self, dataset):
        """
        :param dataset:
        :return:
        """
        (__X_train, __y_train), (__X_test, __y_test) = dataset
        # normalize inputs from 0-255 to 0.0-1.0
        __X_train = __X_train.astype('float32')
        __X_test = __X_test.astype('float32')
        __X_train = __X_train / 255.0
        __X_test = __X_test / 255.0

        # one hot encode outputs
        __y_train = np_utils.to_categorical(__y_train)
        __y_test = np_utils.to_categorical(__y_test)

        # reshape X
        __X_train = numpy.reshape(__X_train, (len(__X_train), 32, 96))
        __X_test = numpy.reshape(__X_test, (len(__X_test), 32, 96))
        return (__X_train, __y_train), (__X_test, __y_test)
