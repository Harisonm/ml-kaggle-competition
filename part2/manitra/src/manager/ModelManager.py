from tensorflow.python.keras.utils import np_utils

import random


class ModelManager(object):

    def __init__(self, param=None, dataset=None):
        self.__param = param
        self.__dataset = dataset
        pass

    @classmethod
    def _build_param(cls, param):
        return {
            'lr': param.get('lr'),
            'optimizer': param.get('optimizer'),
            'losses': param.get('losses'),
            'last_activation': param.get('last_activation'),
            'metrics': param.get('metrics'),
            'epochs': param.get('epochs')
        }

    def _preprocess_i_wild_cam(self):
        pass

    def _run_model(self):
        """
        :return:
        """
        pass

    def _run_model_sample_cnn(self):
        """
        :return:
        """
        pass

    def _save_tensorboard(self, model=None, type_model=None):
        """
        :param model:
        :param type_model:
        :return:
        """
        pass
