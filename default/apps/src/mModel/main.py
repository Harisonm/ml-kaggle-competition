from default.apps.src.mModel.model.modelManager import ModelManager
from default.apps.src.mModel.model.cnn import Cnn
from default.apps.src.mModel.model.mlp import Mlp
from default.apps.src.mModel.model.sp import Sp

from keras.datasets import cifar10
import os
import sys

if __name__ == '__main__':

    # model = sys.argv[1]
    # param = sys.argv[2]
    # Param to MLP
    hyperParamSample = {"activation_1": 'relu',
                      "activation_2": 'softmax',
                      "loss": 'categorical_crossentropy',
                      "optimizer": 'adam',
                      "metrics": ['accuracy'],
                      "input_shape": 3072,
                      "dropout": 0.2,
                      "layerParam": {"denseIn": 1024,
                                     "denseMiddle": 512,
                                     "denseOut": 10}}

    hyperParamCNN = {"activation_1": 'relu',
                      "activation_2": 'softmax',
                      "loss": 'categorical_crossentropy',
                      "padding": 'same',
                      "metrics": ['accuracy'],
                      "input_shape": (3, 32, 32),
                      "dropout": {"param1": 0.2,
                                  "param2": 0.5},
                      "layerParam": {"denseMiddle": 512}}
    nb_epoch = 200
    batch_size = 128
    nb_classes = 10
    dataset = cifar10.load_data()
    # Cnn(hyperParamCNN, nb_epoch, batch_size, nb_classes, dataset).run_model_sample_cnn()
    Mlp(hyperParamSample, nb_epoch, batch_size, nb_classes, dataset).run_model()
    # Sp(hyperParamCNN, nb_epoch, batch_size, nb_classes, dataset).run_model()
