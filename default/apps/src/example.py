import keras
import concurrent.futures
import numpy as np
from keras.optimizers import sgd
from numpy.random import random, randint
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.callbacks import TensorBoard
from keras.layers import Dense, Dropout, Activation, Flatten

THREAD_NUMBER = 10

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

def run_model(model,nb_epochs,batch_size):

    # training Model
    for i in range(100):

        nbLayers = randint(1, 11)
        nbNeuronsPerLayer = randint(8, 256)
        lr = random() * (0.1 - 0.0001) + 0.0001

        # multi threading
        for layer in range(nbLayers):
            if i == layer:
                model.add(Dense(nbNeuronsPerLayer, activation="relu", input_dim=784))
            else:
                model.add(Dense(nbNeuronsPerLayer, activation="tanh"))

        model.add(Dense(10, activation="softmax"))

    # compile Model
    model.compile(optimizer=sgd(lr=lr), loss="mse", metrics=["accuracy"])

    model_str = "mlp" + str(nbLayers) + "_" + str(nbNeuronsPerLayer) + "_" + "softmax_sgd_" + str(lr) + "_mse"

    model.save("./saved_models" + model_str, True, True)
    # Tensorboard callback
    tb_callback = TensorBoard(log_dir="./logs/" + "mlp" + "_784_" + str(i) + "_10_" + "_tanh_softmax_mse")

    model.summary()
    # Training modele
    history = model.fit(X_train, Y_train, 
                batch_size=batch_size,
                validation_data=(X_test, Y_test), 
                epochs=nb_epochs, 
                callbacks=[tb_callback])

    save_history(history, 'history.txt')

    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test acc:', acc)

if __name__ == '__main__':
    # Preprocessing
    (X_train, Y_train), (X_test, Y_test) = keras.datasets.mnist.load_data()

    X_train = np.reshape(X_train, (-1, 784)) / 255.0
    X_test = np.reshape(X_test, (-1, 784)) / 255.0

    Y_train = keras.utils.to_categorical(Y_train, 10)
    Y_test = keras.utils.to_categorical(Y_test, 10)

    # Create model
    model = keras.Sequential()

    with concurrent.futures.ThreadPoolExecutor(THREAD_NUMBER) as executor:

            # Run Thread by Thread by date
            executor.submit(run_model, model)


