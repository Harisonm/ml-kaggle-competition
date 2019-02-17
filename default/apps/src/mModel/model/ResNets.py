from default.apps.src.mModel.manager.ModelManager import ModelManager
from default.apps.src.mModel.manager.LogBuilder import LogBuilder
from tensorflow.python.keras import Input, Model
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.layers import Dense, Add, Dropout
import numpy as np


PATH_TB = "./logsModel/tensorboard/"
PATH_HISTORY = "./logsModel/history/"


class ResNets(ModelManager, LogBuilder):

    def __init__(self, param, dataset):
        """
        :param param:
        :param dataset:
        """
        super().__init__(param, dataset)
        self.__param = self._random_param(param)
        self.__dataset = self._preprocess_cifar10(dataset)
        self.__network_config = self.__network_architecture_builder()
        self.__network_architecture = []

    def run_model(self):
        """
        :return:
        """
        (X_train, y_train), (X_test, y_test) = self.__dataset
        type_model = "resnets"

        # Call function to build network
        self.__network_builder()

        # Call model
        model = self.__network_architecture[-1]

        # Compile model
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

        # Final evaluation of the model
        score = model.evaluate(X_test, y_test, verbose=1)

        print('test loss:', score[0])
        print('test acc:', score[1])

        self._run_ml_flow(self.__param, history, model, score)
        return history, model

    def __network_builder(self):

        while len(self.__network_config) > 0:

            # First Layer
            if 'input_layer' in self.__network_config:

                # Add first layer
                input_layer = Input(shape=(self.__network_config['input_layer']['input_dim'],))

                # Add first hidden Layer
                hidden1 = Dense(self.__network_config['hidden_layer']['hidden_dim'],
                                activation=self.__network_config['hidden_layer']['activation'])(input_layer)

                hidden2 = Dense(self.__network_config['hidden_layer']['hidden_dim'],
                                activation=self.__network_config['hidden_layer']['activation'])(hidden1)

                res = Add()([hidden1, hidden2])

                # store data in list
                self.__network_architecture.append(input_layer)
                self.__network_architecture.append(hidden1)
                self.__network_architecture.append(hidden2)
                self.__network_architecture.append(res)

                # Delete pills of stack
                self.__network_config.pop('input_layer')
                self.__network_builder()

            # Hidden Layer Part
            elif len(self.__network_config) > 1 and self.__network_config['hidden_layer']['nbr_hidden_layer'] > 0:
                hidden = Dense(self.__network_config['hidden_layer']['hidden_dim'],
                               activation=self.__network_config['hidden_layer']['activation'])(self.__network_architecture[-1])

                res = Add()([self.__network_architecture[-2], self.__network_architecture[-1]])

                self.__network_architecture.append(hidden)
                self.__network_architecture.append(res)

                # decrement nbr_hidden_layer
                self.__network_config['hidden_layer']['nbr_hidden_layer'] -= 1

                if self.__network_config['hidden_layer']['nbr_hidden_layer'] == 0:

                    # Delete last layer
                    self.__network_config.pop('hidden_layer')
                self.__network_builder()

            # Last Layer
            elif len(self.__network_config) == 1:
                output_layer = Dense(self.__network_config['output_layer']['output_dim'],
                                     activation=self.__network_config['output_layer']['activation'])(self.__network_architecture[-1])

                self.__network_architecture.append(output_layer)
                self.__network_config.pop('output_layer')
                self.__network_architecture.append(Model(self.__network_architecture[0], output_layer))

    def __network_architecture_builder(self):
        # Configuration to ResNets
        network_conf = {}
        network_conf.update({'input_layer': {'input_dim': self.__param['input_shape']}
                             })

        network_conf.update({'hidden_layer': {'hidden_dim': self.__param['hidden_dim'],
                                              'activation': self.__param['activation'],
                                              'nbr_hidden_layer': self.__param['hidden_layers']
                                              }
                             })

        network_conf.update({'output_layer': {'output_dim': self.__param['unitsSlp'],
                                              'activation': self.__param['last_activation']}
                             })
        return network_conf

    def __save_tensorboard(self, model, type_model):
        """
        :param model:
        :param type_model:
        :return:
        """
        model_str = type_model + "_" + \
                    str(self.__param['hidden_layers']) + "_" + \
                    str(self.__param['epochs']) + "_" + \
                    str(self.__param['batch_size']) + "_" + \
                    str(self.__param['activation']) + "_" + \
                    str(self.__param['losses']) + "_" + \
                    str(self.__param['optimizer'])

        model.save(PATH_TB + type_model + "/saved_models" + "_" + model_str, True, True)

        # Save tensorboard callback
        tb_callback = TensorBoard(log_dir=str(PATH_TB) + "/" + type_model + "/" + type_model + "_" +
                                          str(self.__param['hidden_layers']) + "_" +
                                          str(self.__param['epochs']) + "_" +
                                          str(self.__param['batch_size']) + "_" +
                                          str(self.__param['activation']) + "_" +
                                          str(self.__param['losses']) + "_" +
                                          str(self.__param['optimizer']))
        return tb_callback
