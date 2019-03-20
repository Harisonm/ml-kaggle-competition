from default.apps.src.KerasModel.manager.ModelManager import ModelManager
from default.apps.src.KerasModel.builder.MLFlowBuilder import MLFlowBuilder
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import TensorBoard

PATH_TB = "./logsModel/tensorboard/"
PATH_HISTORY = "./logsModel/history/"


class Mlp(ModelManager, MLFlowBuilder):

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
        type_model = "mlp"

        # MLP
        # A MODIFIER : Mettre une condition pour construire des modèles sans Séquentiel et d'autre avec Sequential
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

        self._run_ml_flow(type_model, self.__param, history, model, score)
        return history, model

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
