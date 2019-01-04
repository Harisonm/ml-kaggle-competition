# Simple CNN model for CIFAR-10
import numpy
import keras
import numpy as np
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard
K.set_image_dim_ordering('th')


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


if __name__ == '__main__':
    nb_epoch = 200
    batch_size = 128
    nb_classes = 10
    
    # load data
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    # normalize inputs from 0-255 to 0.0-1.0
    X_train = X_train.reshape(50000, 32 * 32 * 3)
    X_test = X_test.reshape(10000, 32 * 32 * 3)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    # one hot encode outputs
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # MLP
    model = Sequential()
    model.add(Dense(10, input_shape=(3072, ), activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.2))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    #model.compile(loss='mse', 
    #                optimizer='sgd',
    #                metrics=['accuracy'])
    model.summary()

    #Tensorboard
    experiment_id = "linear_Regression_PS_2_Epochs/"
    tbCallback = keras.callbacks.TensorBoard("./logs/" + experiment_id)

    # Fit the model
    history = model.fit(X_train,y_train, epochs=5000, verbose=0,callbacks = [tbCallback])

    #evaluate the network
    loss, accuracy = model.evaluate(X_train, y_train)
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy*100))

    # training
    history = model.fit(X_train, Y_train,
                        batch_size=batch_size,
                        nb_epoch=nb_epoch,
                        verbose=1,
                        validation_data=(X_test, Y_test))

    save_history(history, 'history.txt')

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test acc:', acc)