from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.constraints import maxnorm
from default.apps.KerasModel.model.Cnn import Cnn
from default.apps.KerasModel.model import Mlp
from default.apps.KerasModel.model import Slp
from default.apps.KerasModel.model.Rnn import Rnn
from default.apps.KerasModel.model import CnnLstm
from default.apps.KerasModel.model.ResNets import ResNets
from numpy.random import random
import sys
import kaggle.api.kaggle_api

if __name__ == '__main__':
    kaggle.KaggleApidatasets

