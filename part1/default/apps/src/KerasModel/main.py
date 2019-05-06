from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.datasets import cifar10
from tensorflow.python.keras.constraints import maxnorm
from default.apps.src.KerasModel.model.Cnn import Cnn
from default.apps.src.KerasModel.model.Mlp import Mlp
from default.apps.src.KerasModel.model.Slp import Slp
from default.apps.src.KerasModel.model.Rnn import Rnn
from default.apps.src.KerasModel.model.CnnLstm import CnnLstm
from default.apps.src.KerasModel.model.ResNets import ResNets
from numpy.random import random
import sys

if __name__ == '__main__':

    type_model = sys.argv[1]
    epochs = int(sys.argv[2])

    lr = random() * (0.1 - 0.0001) + 0.0001
    momentum = random() * (0.1 - 0.0001) + 0.0001
    decay = lr / epochs

    Param = {'input_shape': 3072,
             'input_shape_rnn': (32, 96),
             'input_shape_cnn': (32, (3, 3)),
             'lr': lr,
             'hidden_dim': 128,
             'units': 512,
             'unitsSlp': 10,
             'last_units': 10,
             'first_neuron': [4, 8, 16, 32, 64],
             'hidden_layers': [2, 4, 6, 8, 9, 10, 20, 25, 30],
             'convolution_layer_set': [1, 2, 3],
             'convolution_layer_set_global': [1, 2, 3],
             'filters': [32, 64, 128],
             'pool_size': (2, 2),
             'kernel_constraint': maxnorm(3),
             'weight_decay': 1e-4,
             'batch_size': (64, 128, 512, 1024, 2048),
             'epochs': epochs,
             'dropout': (0, 0.2, 0.4, 0.6, 0.8),
             'padding': 'same',
             'metrics': ['accuracy'],
             'weight_regulizer': [None],
             'emb_output_dims': [None],
             'shape': ['brick', 'long_funnel'],
             'optimizer': ['adam', 'Nadam', 'RMSprop', SGD(lr=lr, momentum=momentum, decay=decay, nesterov=False)],
             'losses': ['mse', 'logcosh', 'categorical_crossentropy'],
             'activation': ['relu', 'linear'],
             'last_activation': ['softmax', 'sigmoid'],
             'nb_classes': 10}

    dataset = cifar10.load_data()
    if type_model == "cnn":
        Cnn(Param, dataset).run_model()
    elif type_model == "mlp":
        Mlp(Param, dataset).run_model()
    elif type_model == "slp":
        Slp(Param, dataset).run_model()
    elif type_model == "rnn":
        Rnn(Param, dataset).run_model()
    elif type_model == "resnets":
        ResNets(Param, dataset).run_model()
    elif type_model == "cnnlstm":
        CnnLstm(Param, dataset).run_model()
