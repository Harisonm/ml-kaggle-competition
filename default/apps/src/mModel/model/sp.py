from default.apps.src.mModel.model.modelManager import ModelManager
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import TensorBoard


class Sp(ModelManager):

    def __init__(self, hyperParameter, nb_epoch, batch_size, nb_classes, dataset):
        ModelManager.__init__(self, hyperParameter, nb_epoch, batch_size, nb_classes, dataset)
        self.__hyperParameter = hyperParameter
        self.__nb_epoch = nb_epoch
        self.__batch_size = batch_size
        self.__nb_classes = nb_classes
        self.__dataset = ModelManager.preprocess(self)

    def run_model(self):
        (X_train, y_train), (X_test, y_test) = self.__dataset
        model = Sequential()
        model.add(Dense(1024, input_shape=(3072,), activation=self.__hyperParameter.get("activation_1")))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation=self.__hyperParameter.get("activation_2")))

        model.compile(loss=self.__hyperParameter.get("loss"),
                      optimizer=self.__hyperParameter.get("optimizer"),
                      metrics=self.__hyperParameter.get("metrics"))
        model.summary()

        # Tensorboard
        # model_str = "_SP_" + "_10_" + "relu_" + "categorical_crossentropy_"

        # model.save("./tensorboard/saved_models" + model_str, True, True)
        # Tensorboard callback
        # tb_callback = TensorBoard(log_dir="./logs/" + "_SP_" + "_1024_" + "_10_" + "_relu_softmax_adam")
        # training

        history = model.fit(X_train, y_train,
                            batch_size=self.__batch_size,
                            epochs=self.__nb_epoch,
                            verbose=1,
                            validation_data=(X_test, y_test))

        self.__save_history(history, 'history.txt')

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', loss)
        print('Test acc:', acc)
