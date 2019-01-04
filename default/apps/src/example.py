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

    # Training modele
    model.fit(X_train, y_train, 
                batch_size=batch_size,
                validation_data=(X_test, y_test), 
                epochs=nb_epochs, 
                callbacks=[tb_callback])


if __name__ == '__main__':
    # Preprocessing
    nb_classes = 10
    nb_epochs = 500
    batch_size = 1024
    
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    X_train = X_train.reshape(50000, 32 * 32 * 3)
    X_test = X_test.reshape(10000, 32 * 32 * 3)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255.0
    X_test /= 255.0

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    # Create model
    model = keras.Sequential()

    with concurrent.futures.ThreadPoolExecutor(THREAD_NUMBER) as executor:

            # Run Thread by Thread by date
            executor.submit(run_model, model , nb_epochs, batch_size)


