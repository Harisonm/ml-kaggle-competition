#from default.apps.src.mModel.mlp.mlp import Mlp
from keras.datasets import cifar10

if __name__ == '__main__':
    hyperParameter = {"activation_1": 'relu',
                      "activation_2": 'softmax',
                      "loss": 'categorical_crossentropy',
                      "optimizer": 'adam',
                      "metrics": ['accuracy']}
    nb_epoch = 200
    batch_size = 128
    nb_classes = 10
    dataset = cifar10.load_data()

    Mlp(hyperParameter).run_model(nb_epoch, batch_size, nb_classes, dataset)
