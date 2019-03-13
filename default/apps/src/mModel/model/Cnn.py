from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import Flatten
from tensorflow.python.keras.layers import Activation
from tensorflow.python.keras.layers import BatchNormalization
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.convolutional import MaxPooling2D
import tensorflow as tf
from tensorflow.python.keras.utils import np_utils
from tensorflow.python.keras.callbacks import TensorBoard
from default.apps.src.mModel.manager.ModelManager import ModelManager
from default.apps.src.mModel.builder.MLFlowBuilder import MLFlowBuilder

tf.keras.backend.backend()
PATH_TB = "./logsModel/tensorboard/"
PATH_HISTORY = "./logsModel/history/"


class Cnn(ModelManager, MLFlowBuilder):

    def __init__(self, param, dataset):
        """
        :param param:
        :param dataset:
        """
        super().__init__(param, dataset)
        self.__param = self._random_param(param)
        self.__dataset = self._preprocess_cifar10(dataset)
        self.__network_config = self.__network_architecture_builder()

    def run_model(self):
        """
        :return:
        """
        # load data
        (X_train, y_train), (X_test, y_test) = self.__dataset
        nb_classes = y_test.shape[1]
        type_model = "cnn"

        # Create the model
        # Mettre une condition pour construire des modèles sans Séquentiel et d'autre avec Sequential
        model = Sequential()

        # Premiere bloc
        for iterator_set_out in range(0, self.__network_config['convolution_layer_set_1']):
            for iterator_set_in in range(0, self.__network_config['convolution_layer_set_2']):

                if iterator_set_out == 0 & iterator_set_in == 0:
                    model.add(Conv2D(32, (3, 3),
                                     padding=self.__param['padding'],
                                     kernel_constraint=self.__param['kernel_constraint'],
                                     kernel_regularizer=regularizers.l2(self.__param['weight_decay']),
                                     input_shape=X_train.shape[1:]))
                else:
                    model.add(Conv2D(32, (3, 3),
                                     padding=self.__param['padding'],
                                     kernel_constraint=self.__param['kernel_constraint']),
                              kernel_regularizer=regularizers.l2(self.__param['weight_decay']))

                model.add(Activation(self.__param['activation']))
                model.add(BatchNormalization())

            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(self.__param['dropout']))

        model.add(Flatten())

        model.add(Dense(self.__param['units'],
                        activation=self.__param['activation'],
                        kernel_constraint=self.__param['kernel_constraint']))

        model.add(Dropout(self.__param['dropout']))

        # Ajout d'un bloc supplémentaire
        # Derniere Couche de fonction
        model.add(Dense(nb_classes,
                        activation=self.__param['activation']))

        model.compile(loss=self.__param['losses'],
                      optimizer=self.__param['optimizer'],
                      metrics=['accuracy'])

        model.summary()

        tb_callback = self.__save_tensorboard(model, type_model)

        # Fit the model
        history = model.fit(X_train, y_train,
                            validation_data=(X_test, y_test),
                            epochs=self.__param['epochs'],
                            batch_size=self.__param['batch_size'],
                            callbacks=[tb_callback])

        # Final evaluation of the model
        score = model.evaluate(X_test, y_test, verbose=1)

        print('test loss:', score[0])
        print('test acc:', score[1])

        self._run_ml_flow(type_model, self.__param, history, model, score)
        return history, model

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

        return (__X_train, __y_train), (__X_test, __y_test)

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

    def __network_architecture_builder(self):
        """
        network_architecture_builder : Configuration of architecture ResNets network
        :return:
        """

        network_conf = {}
        network_conf.update({'convolution_layer_set_1': self.__param['convolution_layer_set']
                             })

        network_conf.update({'convolution_layer_set_2': self.__param['convolution_layer_set_global']
                             })
        return network_conf
