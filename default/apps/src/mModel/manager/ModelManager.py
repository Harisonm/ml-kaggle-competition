from keras.utils import np_utils
import random
from keras.constraints import maxnorm


class ModelManager(object):

    def __init__(self, param=None, dataset=None):
        self.__param = param
        self.__dataset = dataset
        pass

    @classmethod
    def _random_param(cls, param):
        rand_param = {'input_shape': param['input_shape'],
                      'input_shape_cnn': param['input_shape_cnn'],
                      'lr': param['lr'],
                      'units': param['units'],
                      'unitsSlp': param['unitsSlp'],
                      'padding': param['padding'],
                      'last_units': param['last_units'],
                      'first_neuron': random.choice(param['first_neuron']),
                      'hidden_layers': random.choice(param['hidden_layers']),
                      'kernel_constraint': param['kernel_constraint'],
                      'batch_size': random.choice(param['batch_size']),
                      'epochs': random.choice(param['epochs']),
                      'dropout': random.choice(param['dropout']),
                      'metrics': param['metrics'],
                      'weight_regulizer': param['weight_regulizer'],
                      'emb_output_dims': ['emb_output_dims'],
                      'shape': random.choice(param['shape']),
                      'optimizer': random.choice(param['optimizer']),
                      'losses': random.choice(param['losses']),
                      'activation': random.choice(param['activation']),
                      'last_activation': random.choice(param['last_activation']),
                      'nb_classes': param['nb_classes']}
        return rand_param

    @staticmethod
    def _save_history(history, result_file):

        loss = history.history['loss']
        acc = history.history['acc']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_acc']
        nb_epoch = len(acc)

        with open(result_file, "w") as fp:
            fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
            for i in range(nb_epoch):
                fp.write("%d\t%f\t%f\t%f\t%f\n" %
                         (i, loss[i], acc[i], val_loss[i], val_acc[i]))

    def _preprocess_cifar10(self, dataset):
        '''
        :return:
        '''
        (__X_train, __y_train), (__X_test, __y_test) = dataset
        __X_train = __X_train.reshape(50000, 32 * 32 * 3)
        __X_test = __X_test.reshape(10000, 32 * 32 * 3)

        __X_train = __X_train.astype('float32')
        __X_test = __X_test.astype('float32')
        __X_train /= 255.0
        __X_test /= 255.0

        __y_train = np_utils.to_categorical(__y_train, self.__param['nb_classes'])
        __y_test = np_utils.to_categorical(__y_test, self.__param['nb_classes'])
        return (__X_train, __y_train), (__X_test, __y_test)

    def _run_model(self):
        '''
        :return:
        '''
        pass

    def _run_model_sample_cnn(self):
        '''
        :return:
        '''
        pass

    def _save_tensorboard(self, model=None, type_model=None):
        """
        :param model:
        :param type_model:
        :return:
        """
        pass
