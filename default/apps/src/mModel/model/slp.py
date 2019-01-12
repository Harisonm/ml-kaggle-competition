from default.apps.src.mModel.model.modelManager import ModelManager
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import TensorBoard


class Slp(ModelManager):

    def __init__(self, hyperParameter, nb_epoch, batch_size, nb_classes, dataset):
        '''
        :param hyperParameter:
        :param nb_epoch:
        :param batch_size:
        :param nb_classes:
        :param dataset:
        '''
        ModelManager.__init__(self, hyperParameter, nb_epoch, batch_size, nb_classes, dataset)
        self.__hyperParameter = hyperParameter
        self.__nb_epoch = nb_epoch
        self.__batch_size = batch_size
        self.__nb_classes = nb_classes
        self.__dataset = ModelManager.preprocess(self)

    def run_model(self):
        '''
        :return:
        '''
        (X_train, y_train), (X_test, y_test) = self.__dataset
        model = Sequential()
        model.add(Dense(self.__hyperParameter.get("layerParam").get("denseIn"),
                        input_shape=(self.__hyperParameter.get("input_shape"),),
                        activation=self.__hyperParameter.get("activation_1")))

        model.add(Dropout(self.__hyperParameter.get("dropout")))

        model.add(Dense(self.__hyperParameter.get("layerParam").get("denseOut"),
                        activation=self.__hyperParameter.get("activation_2")))

        model.compile(loss=self.__hyperParameter.get("loss"),
                      optimizer=self.__hyperParameter.get("optimizer"),
                      metrics=self.__hyperParameter.get("metrics"))
        model.summary()

        type_model = "slp1"
        tb_callback = super().save_tensorboard(model, type_model)
        # training
        history = model.fit(X_train, y_train,
                            batch_size=self.__batch_size,
                            epochs=self.__nb_epoch,
                            verbose=1,
                            validation_data=(X_test, y_test),
                            callbacks=[tb_callback])

        self.__save_history(history, 'history.txt')

        loss, acc = model.evaluate(X_test, y_test, verbose=1)
        print('Test loss:', loss)
        print('Test acc:', acc)

