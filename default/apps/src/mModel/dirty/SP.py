# import talos
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, Nadam
from keras.losses import categorical_crossentropy, logcosh
from keras.activations import relu, elu, softmax
from keras.callbacks import TensorBoard


def save_history(history, result_file):
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


def run_model(model, nb_epochs, batch_size):
    model.add(Dense(1024, input_shape=(3072, ), activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation='softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Tensorboard
    model_str = "_SP_" + "_10_" + "relu_" + "categorical_crossentropy_"

    model.save("./tensorboard/saved_models" + model_str, True, True)
    # Tensorboard callback
    tb_callback = TensorBoard(log_dir="./logs/" + "_SP_" + "_1024_" + "_10_" + "_relu_softmax_adam")
    
    model.summary()
    
    # training
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        epochs=nb_epochs,
                        verbose=1,
                        validation_data=(X_test, Y_test),
                        callbacks=[tb_callback])

    save_history(history, 'history' + '_SP_' + '.txt')
 
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test acc:', acc)


if __name__ == '__main__':
    #Param
    nb_epoch = 1000
    batch_size = 2048
    nb_classes = 10

    # Preprocessing
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.reshape(50000, 32 * 32 * 3)
    X_test = X_test.reshape(10000, 32 * 32 * 3)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # SP
    model = Sequential()
    run_model(model, nb_epoch, batch_size)