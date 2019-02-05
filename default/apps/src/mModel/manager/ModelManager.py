from keras.utils import np_utils
from mlflow import log_metric, log_param, log_artifact
from keras.constraints import maxnorm
import os
import sys
import random
import mlflow
import mlflow.keras


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
        acc = history.history['binary_accuracy']
        val_loss = history.history['val_loss']
        val_acc = history.history['val_binary_accuracy']
        nb_epoch = len(acc)

        with open(result_file, "w") as fp:
            fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
            for i in range(nb_epoch):
                fp.write("%d\t%f\t%f\t%f\t%f\n" %
                         (i, loss[i], acc[i], val_loss[i], val_acc[i]))

    def _preprocess_cifar10(self, dataset):
        """
        :param dataset:
        :return:
        """
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

    @staticmethod
    def _get_binary_loss(hist):
        """
        :param hist:
        :return:
        """
        loss = hist.history['loss']
        loss_val = loss[len(loss) - 1]
        return loss_val

    @staticmethod
    def _get_binary_acc(hist):
        """
        :param hist:
        :return:
        """
        acc = hist.history['binary_accuracy']
        acc_value = acc[len(acc) - 1]
        return acc_value

    @staticmethod
    def _get_validation_loss(hist):
        """
        :param hist:
        :return:
        """
        val_loss = hist.history['val_loss']
        val_loss_value = val_loss[len(val_loss) - 1]
        return val_loss_value

    @staticmethod
    def _get_validation_acc(hist):
        """
        :param hist:
        :return:
        """
        val_acc = hist.history['val_binary_accuracy']
        val_acc_value = val_acc[len(val_acc) - 1]
        return val_acc_value
    
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

    def _get_directory_path(self, dir_name, create_dir=True):

        cwd = os.getcwd()
        dir = os.path.join(cwd, dir_name)
        if create_dir:
            if not os.path.exists(dir):
                os.mkdir(dir, mode=0o755)

        return dir

    def __get_directory_path(self, dir_name, create_dir=True):
        cwd = os.getcwd()
        dir = os.path.join(cwd, dir_name)
        if create_dir:
            if not os.path.exists(dir):
                os.mkdir(dir, mode=0o755)
        return dir

    def _run_ml_flow(self, history, model, score):
        with mlflow.start_run():
            # print out current run_uuid
            run_uuid = mlflow.active_run().info.run_uuid
            print("MLflow Run ID: %s" % run_uuid)

            # log parameters
            mlflow.log_param("input_shape", self.__param['input_shape'])
            mlflow.log_param("activation", self.__param['activation'])
            mlflow.log_param("epochs", self.__param['epochs'])
            mlflow.log_param("loss_function", self.__param['losses'])
            mlflow.log_param("last_activation", self.__param['last_activation'])
            mlflow.log_param("optimizer", self.__param['optimizer'])
            mlflow.log_param("lr", self.__param['lr'])

            # calculate metrics
            binary_loss = self._get_binary_loss(history)
            binary_acc = self._get_binary_acc(history)
            validation_loss = self._get_validation_loss(history)
            validation_acc = self._get_validation_acc(history)
            average_loss = score[0]
            average_acc = score[1]

            # log metrics
            mlflow.log_metric("binary_loss", binary_loss)
            mlflow.log_metric("binary_acc", binary_acc)
            mlflow.log_metric("validation_loss", validation_loss)
            mlflow.log_metric("validation_acc", validation_acc)
            mlflow.log_metric("average_loss", average_loss)
            mlflow.log_metric("average_acc", average_acc)

            # log artifacts (matplotlib images for loss/accuracy)
            # log model
            mlflow.keras.log_model(model, "logsModel/models")
            image_dir = self.__get_directory_path("logsModel/images")
            # log artifacts
            mlflow.log_artifacts(image_dir, "logsModel/images")

            # save model locally
            pathdir = "keras_models/" + run_uuid

            # Write out TensorFlow events as a run artifact
            print("Uploading TensorFlow events as a run artifact.")
            #mlflow.log_artifacts(output_dir, artifact_path="events")
            mlflow.end_run()

        print("loss function use", self.__param['losses'])
        pass
