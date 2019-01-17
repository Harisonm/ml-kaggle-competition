from default.apps.src.mModel.model.modelManager import ModelManager
from default.apps.src.mModel.model.cnn import Cnn
from default.apps.src.mModel.model.mlp import Mlp
from default.apps.src.mModel.model.slp import Slp
from keras.optimizers import SGD
from keras.datasets import cifar10
from keras.losses import mse
import os
import sys
from keras.constraints import maxnorm

if __name__ == '__main__':
    # model = sys.argv[1]
    # param = sys.argv[2]
    # Param to MLP

    hyperParamSample = {'activation': 'linear',
                        'optimizer': SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False),
                        'input_shape': 3072,
                        'metrics': ['accuracy'],
                        'loss': [mse],
                        'units': 10}

    # hyperParamCNN = {"activation_1": 'relu',
    #                  "activation_2": 'softmax',
    #                  "loss": 'categorical_crossentropy',
    #                  "padding": 'same',
    #                  "metrics": ['accuracy'],
    #                  "input_shape": (3, 32, 32),
    #                  "dropout": {"param1": 0.2,
    #                              "param2": 0.5},
    #                  "layerParam": {"denseMiddle": 512}}

    # hyperParamTmp = {'input_shape': 3072,
    #                     'lr': (0.5, 5, 10),
    #                     'units': 1024,
    #                     'last_units': 10,
    #                     'first_neuron': [4, 8, 16, 32, 64],
    #                     'hidden_layers': [0, 1, 2],
    #                     'kernel_constraint': maxnorm(3),
    #                     'batch_size': (2, 30, 10),
    #                     'epochs': [150],
    #                     'dropout': (0, 0.5, 5, 1),
    #                     'metrics': ['accuracy'],
    #                     'weight_regulizer': [None],
    #                     'emb_output_dims': [None],
    #                     'shape': ['brick', 'long_funnel'],
    #                     'optimizer': [adam, Nadam, RMSprop, SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)],
    #                     'losses': [logcosh, binary_crossentropy, categorical_crossentropy],
    #                     'activation': [relu, elu],
    #                     'last_activation': [softmax, sigmoid]}
    nb_epoch = 100
    batch_size = 1024
    nb_classes = 10
    dataset = cifar10.load_data()
    # Cnn(hyperParamCNN, nb_epoch, batch_size, nb_classes, dataset).run_model_sample_cnn()
    # Mlp(hyperParamSample, nb_epoch, batch_size, nb_classes, dataset).run_model()
    Slp(hyperParamSample, nb_epoch, batch_size, nb_classes, dataset).run_model()
