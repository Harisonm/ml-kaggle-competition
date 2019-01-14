from default.apps.src.mModel.model.modelManager import ModelManager
from default.apps.src.mModel.model.cnn import Cnn
from default.apps.src.mModel.model.mlp import Mlp
from default.apps.src.mModel.model.slp import Slp
from keras.optimizers import SGD, adam, Nadam, RMSprop
from keras.losses import logcosh, binary_crossentropy, categorical_crossentropy
from keras.activations import softmax, sigmoid, relu, elu
from keras.datasets import cifar10
import os
import sys
from keras.constraints import maxnorm

if __name__ == '__main__':
    # model = sys.argv[1]
    # param = sys.argv[2]
    # Param to MLP
    # hyperParamSample = {'activation_1': 'relu',
    #                     'activation_2': 'softmax',
    #                     'loss': 'categorical_crossentropy',
    #                     "optimizer": {"adam": 'adam',
    #                                   "sgd": SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)},
    #                     'metrics': ['accuracy'],
    #                     'input_shape': 3072,
    #                     'kernel_constraint': maxnorm(3),
    #                     'dropout': 0.2,
    #                     'layerParam': {'denseIn': 1024,
    #                                    'denseMiddle': 512,
    #                                    'denseOut': 10}}

    # hyperParamSample = {'activation_1': 'relu',
    #                     'activation_2': 'softmax',
    #                     'loss': 'categorical_crossentropy',
    #                     'optimizer': {"adam": 'adam',
    #                                   },
    #                     'metrics': ['accuracy'],
    #                     'input_shape': 3072,
    #                     'kernel_constraint': maxnorm(3),
    #                     'dropout': 0.2,
    #                     'layerParam': {"denseIn": 1024,
    #                                    "denseMiddle": 512,
    #                                    "denseOut": 10}}
    hyperParamSample = {'input_shape': 3072,
                        'lr': (0.5, 5, 10),
                        'units': 1024,
                        'last_units': 10,
                        'first_neuron': [4, 8, 16, 32, 64],
                        'hidden_layers': [0, 1, 2],
                        'kernel_constraint': maxnorm(3),
                        'batch_size': (2, 30, 10),
                        'epochs': [150],
                        'dropout': (0, 0.5, 5, 1),
                        'metrics': ['accuracy'],
                        'weight_regulizer': [None],
                        'emb_output_dims': [None],
                        'shape': ['brick', 'long_funnel'],
                        'optimizer': [adam, Nadam, RMSprop, SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)],
                        'losses': [logcosh, binary_crossentropy, categorical_crossentropy],
                        'activation': [relu, elu],
                        'last_activation': [softmax, sigmoid]}

    # hyperParamCNN = {'activation': relu,
    #                      'last_activation': softmax,
    #                      'losses': categorical_crossentropy,
    #                      'padding': "same",
    #                      "metrics": ['accuracy'],
    #                      'input_shape': (3, 32, 32),
    #                      'dropout': [0.2, 0.5]}
    nb_epoch = 2
    batch_size = 4086
    nb_classes = 10
    dataset = cifar10.load_data()
    Mlp(hyperParamSample, nb_epoch, batch_size, nb_classes, dataset).run_model()
    # if model == "Mlp":
    #     Mlp(hyperParamSample, nb_epoch, batch_size, nb_classes, dataset).run_model()
    # elif model == "Cnn":
    #     Cnn(hyperParamCNN, nb_epoch, batch_size, nb_classes, dataset).run_model_sample_cnn()
    # elif model == "slp":
    #     Slp(hyperParamSample, nb_epoch, batch_size, nb_classes, dataset).run_model()
