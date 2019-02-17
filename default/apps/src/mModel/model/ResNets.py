from default.apps.src.mModel.manager.ModelManager import ModelManager
from default.apps.src.mModel.manager.LogBuilder import LogBuilder
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import TensorBoard


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

    def run_model(self):
        """
        :return:
        """
        (X_train, y_train), (X_test, y_test) = self.__dataset
        type_model = "resnets"

    # input_dim: int, hidden_dim: int, output_dim:int --> param[]

    # input_dim: int, hidden_dim: int, output_dim:int --> param[]

    def network_builder(network_conf, network_archi):

        print("len", len(network_conf))

        while len(network_conf) > 0:

            # Premiere couche
            if 'input_layer' in network_conf:

                # Add first layer
                input_layer = Input(shape=(network_conf['input_layer']['input_dim'],))

                # Add first hidden Layer
                hidden1 = Dense(network_conf['hidden_layer']['hidden_dim'],
                                activation=network_conf['hidden_layer']['activation'])(input_layer)

                hidden2 = Dense(network_conf['hidden_layer']['hidden_dim'],
                                activation=network_conf['hidden_layer']['activation'])(hidden1)

                res = Add()([hidden1, hidden2])

                # store data in list
                network_archi.append(input_layer)
                network_archi.append(hidden1)
                network_archi.append(hidden2)
                network_archi.append(res)

                #### A VIRER
                print('\n'.join(map(str, network_archi)))

                # Delete pills of stack
                network_conf.pop('input_layer')
                network_builder(network_conf, network_archi)

            # Couche caché
            elif len(network_conf) > 1 and network_conf['hidden_layer']['nbr_hidden_layer'] > 0:
                hidden = Dense(network_conf['hidden_layer']['hidden_dim'],
                               activation=network_conf['hidden_layer']['activation'])(network_archi[-1])

                #### A VIRER
                print('AFFICHAGE', network_archi[-2])
                res = Add()([network_archi[-2], network_archi[-1]])

                network_archi.append(hidden)
                network_archi.append(res)

                #### A VIRER
                print('\n'.join(map(str, network_archi)))

                # decrement nbr_hidden_layer
                network_conf['hidden_layer']['nbr_hidden_layer'] -= 1

                if (network_conf['hidden_layer']['nbr_hidden_layer'] == 0):
                    # Derniere couche
                    network_conf.pop('hidden_layer')

                network_builder(network_conf, network_archi)

            elif len(network_conf) == 1:
                output_layer = Dense(network_conf['output_layer']['output_dim'],
                                     activation=network_conf['output_layer']['activation'])(network_archi[-1])
                model_functional = Model(network_archi[0], output_layer)

                #### A VIRER
                print('AFFICHAGE MODEL', model_functional)
                # print de test
                network_archi.append(output_layer)

                #### A VIRER
                print('\n'.join(map(str, network_archi)))

                network_conf.pop('output_layer')

                return model_functional

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
