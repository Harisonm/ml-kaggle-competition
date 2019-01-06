from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils


class Mlp(object):

    def __init__(self, hyperParameter):
        self.__hyperParameter = hyperParameter

    @staticmethod
    def __save_history(history, result_file):
        loss = history.history['loss']
        acc = history.history['acc']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']
        nb_epoch = len(acc)

        with open(result_file, "w") as fp:
            fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
            for i in range(nb_epoch):
                fp.write("%d\t%f\t%f\t%f\t%f\n" %
                         (i, loss[i], acc[i], val_loss[i], val_acc[i]))

    @staticmethod
    def __preprocessing(nb_classes, dataset):
        (__X_train, __y_train), (__X_test, __y_test) = dataset
        __X_train = __X_train.reshape(50000, 32 * 32 * 3)
        __X_test = __X_test.reshape(10000, 32 * 32 * 3)

        __X_train = __X_train.astype('float32')
        __X_test = __X_test.astype('float32')
        __X_train /= 255.0
        __X_test /= 255.0

        __y_train = np_utils.to_categorical(__y_train, nb_classes)
        __y_test = np_utils.to_categorical(__y_test, nb_classes)
        return (__X_train, __y_train), (__X_test, __y_test)

    def run_model(self, nb_epoch, batch_size, nb_classes, dataset):

        (X_train, y_train), (X_test, y_test) = self.__preprocessing(nb_classes, dataset)

        # MLP
        model = Sequential()
        model.add(Dense(1024, input_shape=(3072,), activation=self.__hyperParameter.get("activation_1")))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation=self.__hyperParameter.get("activation_1")))
        model.add(Dropout(0.2))
        model.add(Dense(512, activation=self.__hyperParameter.get("activation_1")))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation=self.__hyperParameter.get("activation_2")))

        model.compile(loss=self.__hyperParameter.get("loss"),
                      optimizer=self.__hyperParameter.get("optimizer"),
                      metrics=self.__hyperParameter.get("metrics"))
        model.summary()

        # training
        history = model.fit(X_train, y_train,
                            batch_size=batch_size,
                            epochs=nb_epoch,
                            verbose=1,
                            validation_data=(X_test, y_test))

        self.__save_history(history, 'history.txt')

        loss, acc = model.evaluate(X_test, y_test, verbose=0)
        print('Test loss:', loss)
        print('Test acc:', acc)
