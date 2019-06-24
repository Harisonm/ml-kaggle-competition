from part2.manitra.src.PreModelCNN import PreModelCNN
from tensorflow.python.keras.optimizers import SGD
from numpy.random import random

import sys

if __name__ == '__main__':
    epochs = int(sys.argv[1])

    lr = random() * (0.1 - 0.0001) + 0.0001
    momentum = random() * (0.1 - 0.0001) + 0.0001
    decay = lr / epochs

    Param = {'lr': lr,
             'optimizer': SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False),
             'losses': 'categorical_crossentropy',
             'last_activation': 'softmax',
             'metrics': 'acc',
             'epochs': epochs,
             }

    PreModelCNN(Param).run_model()
