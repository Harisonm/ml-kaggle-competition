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
            cls._draw_plot_metrics(history, 'binary_loss', 'loss')
            cls._draw_plot_metrics(history, 'accuracy', 'acc')
            cls._draw_plot_metrics(history, 'validation_loss', 'val_loss')
            cls._draw_plot_metrics(history, 'validation_acc', 'val_acc')
            average_loss = score[0]
            average_acc = score[1]

            # log metrics
            mlflow.log_metric("average_loss", average_loss)
            mlflow.log_metric("average_acc", average_acc)

            # save model locally
            path_dir = "keras_models/" + run_uuid

            # log model
            mlflow.keras.log_model(model, "logsModel/models")

            # log artifacts
            path_dir = cls._get_path_folder("logsModel/artifacts")
            mlflow.log_artifacts(path_dir, "logsModel/artifacts")

            # Write out TensorFlow events as a run artifact
            print("Uploading TensorFlow events as a run artifact.")

            # workflow.log_artifacts(output_dir, artifact_path="events")
            mlflow.end_run()

        print("loss function use", param['losses'])
        pass

    @staticmethod
    def _draw_plot_metrics(hist, metrics_name, index):
        """
        :param hist:
        :param metrics_name:
        :param index:
        :return:
        """
        for iterator in hist.history[index]:
            mlflow.log_metric(metrics_name, iterator)

    @staticmethod
    def _get_binary_loss(hist, metrics_name, index):
        """
        :param hist:
        :param metrics_name:
        :param index:
        :return:
        """
        loss = hist.history[index]
        loss_val = loss[len(metrics_name) - 1]
        return loss_val

    @staticmethod
    def _get_path_folder(dir_name, create_dir=True):
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

