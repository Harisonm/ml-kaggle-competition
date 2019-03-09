import mlflow
import mlflow.keras
from tensorflow.python import keras
import os
import sys


class MLFlowBuilder(object):

    @classmethod
    def _run_ml_flow(cls, type_model, param, history, model, score):
        """
        :param param:
        :param history:
        :param model:
        :param score:
        :return:
        """
        experiment_id = mlflow.set_experiment(type_model)

        with mlflow.start_run(experiment_id=experiment_id):
            # print out current run_uuid
            run_uuid = mlflow.active_run().info.run_uuid
            print("MLflow Run ID: %s" % run_uuid)

            # log parameters
            mlflow.log_param("input_shape", param['input_shape'])
            mlflow.log_param("activation", param['activation'])
            mlflow.log_param("epochs", param['epochs'])
            mlflow.log_param("loss_function", param['losses'])
            mlflow.log_param("last_activation", param['last_activation'])
            mlflow.log_param("optimizer", param['optimizer'])
            mlflow.log_param("lr", param['lr'])

            # calculate metrics
            binary_loss = cls._get_binary_loss(history)
            accuracy = cls._get_binary_acc(history)
            validation_loss = cls._get_validation_loss(history)
            validation_acc = cls._get_validation_acc(history)
            average_loss = score[0]
            average_acc = score[1]

            # log metrics
            mlflow.log_metric("binary_loss", binary_loss)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("validation_loss", validation_loss)
            mlflow.log_metric("validation_acc", validation_acc)
            mlflow.log_metric("average_loss", average_loss)
            mlflow.log_metric("average_acc", average_acc)

            # log artifacts (matplotlib images for loss/accuracy)
            # log model
            mlflow.keras.log_model(model, "logsModel/models")
            image_dir = cls._get_dir_path("logsModel/images")
            # log artifacts
            mlflow.log_artifacts(image_dir, "logsModel/images")

            # save model locally
            pathdir = "keras_models/" + run_uuid

            # Write out TensorFlow events as a run artifact
            print("Uploading TensorFlow events as a run artifact.")

            # workflow.log_artifacts(output_dir, artifact_path="events")
            mlflow.end_run()

        print("loss function use", param['losses'])
        pass

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
        acc = hist.history['acc']
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
        val_acc = hist.history['val_acc']
        val_acc_value = val_acc[len(val_acc) - 1]
        return val_acc_value

    @staticmethod
    def _get_dir_path(dir_name, create_dir=True):
        """
        :param dir_name:
        :param create_dir:
        :return:
        """
        cwd = os.getcwd()
        dir_path = os.path.join(cwd, dir_name)
        if create_dir:
            if not os.path.exists(dir_path):
                os.mkdir(dir_path, mode=0o755)
        return dir_path

