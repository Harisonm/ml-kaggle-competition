from keras.utils import np_utils
from keras.callbacks import TensorBoard


class ModelManager(object):

    def __init__(self, hyperParameter, nb_epoch, batch_size, nb_classes, dataset):
        '''
        :param hyperParameter:
        :param nb_epoch:
        :param batch_size:
        :param nb_classes:
        :param dataset:
        '''
        self.__hyperParameter = hyperParameter
        self.__nb_epoch = nb_epoch
        self.__batch_size = batch_size
        self.__nb_classes = nb_classes
        self.__dataset = dataset

    @staticmethod
    def save_history(history, result_file):
        '''
        :param history:
        :param result_file:
        :return:
        '''
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

    def preprocess(self):
        '''
        :return:
        '''
        (__X_train, __y_train), (__X_test, __y_test) = self.__dataset
        __X_train = __X_train.reshape(50000, 32 * 32 * 3)
        __X_test = __X_test.reshape(10000, 32 * 32 * 3)

        __X_train = __X_train.astype('float32')
        __X_test = __X_test.astype('float32')
        __X_train /= 255.0
        __X_test /= 255.0

        __y_train = np_utils.to_categorical(__y_train, self.__nb_classes)
        __y_test = np_utils.to_categorical(__y_test, self.__nb_classes)
        return (__X_train, __y_train), (__X_test, __y_test)

    def run_model(self):
        '''
        :return:
        '''
        pass

    def run_model_sample_cnn(self):
        '''
        :return:
        '''
        pass

    def save_tensorboard(self, model, type_model):
        '''
        :param model:
        :param type_model:
        :return:
        '''
        model_str = type_model + "_" + \
                    str(self.__hyperParameter.get("layerParam").get("denseIn")) + "_" + \
                    str(self.__hyperParameter.get("layerParam").get("denseOut")) + "_" + \
                    str(self.__hyperParameter.get("activation_2")) + "_" + \
                    str(self.__hyperParameter.get("loss")) + "_" + \
                    str(self.__hyperParameter.get("optimizer"))

        model.save("./tensorboard/saved_models" + "_" + model_str, True, True)

        # Save tensorboard callback
        tb_callback = TensorBoard(log_dir="./tensorboard/logs/" + type_model + "_" +
                                          str(self.__hyperParameter.get("layerParam").get("denseIn")) + "_" +
                                          str(self.__hyperParameter.get("layerParam").get("denseOut")) + "_" +
                                          str(self.__hyperParameter.get("loss")) + "_" +
                                          str(self.__hyperParameter.get("optimizer")))
        return tb_callback
