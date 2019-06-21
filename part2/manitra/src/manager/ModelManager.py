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
            'batch_size': param.get('batch_size'),
            'epochs': param.get('epochs'),
            'metrics': param.get('metrics'),
            'losses': param.get('losses'),
            'activation': param.get('activation'),
            'last_activation': param.get('last_activation')
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
