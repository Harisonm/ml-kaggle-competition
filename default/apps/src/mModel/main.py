from default.apps.src.mModel.model.modelManager import ModelManager
from default.apps.src.mModel.model.cnn import Cnn
from default.apps.src.mModel.model.mlp import Mlp
from default.apps.src.mModel.model.slp import Slp
from keras.optimizers import SGD
from keras.datasets import cifar10
import os
import sys
from keras.constraints import maxnorm

if __name__ == '__main__':
    # model = sys.argv[1]
    # param = sys.argv[2]
    # Param to MLP
    hyperParamSample = {"activation_1": 'relu',
                        "activation_2": 'softmax',
                        "loss": 'categorical_crossentropy',
                        "optimizer": {"adam": 'adam',
                                      "sgd": SGD(lr=0.1, momentum=0.9, decay=0.0, nesterov=False)},
                        "metrics": ['accuracy'],
                        "input_shape": 3072,
                        "kernel_constraint": maxnorm(3),
                        "dropout": 0.2,
                        "layerParam": {"denseIn": 1024,
                                       "denseMiddle": 512,
                                       "denseOut": 10}}

    # hyperParamCNN = {"activation_1": 'relu',
    #     #                  "activation_2": 'softmax',
    #     #                  "loss": 'categorical_crossentropy',
    #     #                  "padding": 'same',
    #     #                  "metrics": ['accuracy'],
    #     #                  "input_shape": (3, 32, 32),
    #     #                  "dropout": {"param1": 0.2,
    #     #                              "param2": 0.5},
    #     #                  "layerParam": {"denseMiddle": 512}}
    nb_epoch = 1000
    batch_size = 4086
    nb_classes = 10
    dataset = cifar10.load_data()
    # Cnn(hyperParamCNN, nb_epoch, batch_size, nb_classes, dataset).run_model_sample_cnn()
    Mlp(hyperParamSample, nb_epoch, batch_size, nb_classes, dataset).run_model()
    # Slp(hyperParamCNN, nb_epoch, batch_size, nb_classes, dataset).run_model()
