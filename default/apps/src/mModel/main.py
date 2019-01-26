from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.losses import mse, logcosh, binary_crossentropy, categorical_crossentropy
from keras.activations import relu, elu, softmax, sigmoid, linear
from keras.constraints import maxnorm
from default.apps.src.mModel.model.Cnn import Cnn
from default.apps.src.mModel.model.Mlp import Mlp
from default.apps.src.mModel.model.Slp import Slp
from numpy.random import random
import sys

if __name__ == '__main__':

    type_model = sys.argv[1]

    # Param to MLP
    lr = random() * (0.1 - 0.0001) + 0.0001
    momentum = random() * (0.1 - 0.0001) + 0.0001

    Param = {'input_shape': 3072,
             'input_shape_cnn': (3, 32, 32),
             'lr': lr,
             'units': 512,
             'unitsSlp': 10,
             'last_units': 10,
             'first_neuron': [4, 8, 16, 32, 64],
             'hidden_layers': [2, 4, 6, 8, 9, 10, 20, 25, 30],
             'kernel_constraint': maxnorm(3),
             'batch_size': (1024, 2048),
             'epochs': [100, 500, 1000],
             'dropout': (0, 0.5, 5, 1),
             'padding': 'same',
             'metrics': ['accuracy'],
             'weight_regulizer': [None],
             'emb_output_dims': [None],
             'shape': ['brick', 'long_funnel'],
             'optimizer': ['adam', 'Nadam', 'RMSprop', SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=False)],
             'losses': [mse, logcosh, binary_crossentropy, categorical_crossentropy],
             'activation': [relu, elu, linear],
             'last_activation': [softmax, sigmoid],
             'nb_classes': 10}

    dataset = cifar10.load_data()
    if type_model == "cnn":
        Cnn(Param, dataset).run_model()
    elif type_model == "mlp":
        Mlp(Param, dataset).run_model()
    elif type_model == "slp":
        Slp(Param, dataset).run_model()
