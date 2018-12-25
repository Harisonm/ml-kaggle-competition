import keras
from keras.callbacks import TensorBoard
import numpy as np
from keras.optimizers import sgd
from numpy.random import random, randint
import concurrent.futures

THREAD_NUMBER = 10


def run_model(model):

    # training Model
    for i in range(100):

        nbLayers = randint(1, 11)
        nbNeuronsPerLayer = randint(8, 256)
        lr = random() * (0.1 - 0.0001) + 0.0001

        # multi threading
        for layer in range(nbLayers):
            if i == layer:
                model.add(keras.layers.Dense(nbNeuronsPerLayer, activation="tanh", input_dim=784))
            else:
                model.add(keras.layers.Dense(nbNeuronsPerLayer, activation="tanh"))

        model.add(keras.layers.Dense(10, activation="softmax"))

    # compile Model
    model.compile(optimizer=sgd(lr=lr), loss="mse", metrics=["accuracy"])

    model_str = "mlp" + str(nbLayers) + "_" + str(nbNeuronsPerLayer) + "_" + "softmax_sgd_" + str(lr) + "_mse"

    model.save("./saved_models" + model_str, True, True)
    # Tensorboard callback
    tb_callback = TensorBoard(log_dir="./logs/" + "mlp_1" + "_784_" + str(i) + "_10_" + "_tanh_softmax_mse")

    # Training modele
    model.fit(xtrain, ytrain, validation_data=(xtest, ytest), epochs=100, callbacks=[tb_callback])


if __name__ == '__main__':
    # Preprocessing
    (xtrain, ytrain), (xtest, ytest) = keras.datasets.mnist.load_data()

    xtrain = np.reshape(xtrain, (-1, 784)) / 255.0
    xtest = np.reshape(xtest, (-1, 784)) / 255.0

    ytrain = keras.utils.to_categorical(ytrain, 10)
    ytest = keras.utils.to_categorical(ytest, 10)

    # Create model
    model = keras.Sequential()

    with concurrent.futures.ThreadPoolExecutor(THREAD_NUMBER) as executor:

            # Run Thread by Thread by date
            executor.submit(run_model, model)


