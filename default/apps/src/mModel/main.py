from default.apps.src.mModel.model.modelManager import modelManager
from keras.datasets import cifar10
import os
import sys

if __name__ == '__main__':

    #model = sys.argv[1]
    #param = sys.argv[2]

    hyperParameter = {"activation_1": 'relu',
                      "activation_2": 'softmax',
                      "loss": 'categorical_crossentropy',
                      "optimizer": 'adam',
                      "metrics": ['accuracy']}
    nb_epoch = 200
    batch_size = 128
    nb_classes = 10
    dataset = cifar10.load_data()
    modelManager(hyperParameter, nb_epoch,batch_size,nb_classes, dataset).run_model_mlp()
