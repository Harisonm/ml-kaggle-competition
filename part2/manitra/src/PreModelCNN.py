import json
import pandas as pd
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import LeakyReLU
import os

PATH = 'dataset/'

if __name__ == '__main__':
    train_df = pd.read_csv(os.path.join(PATH, 'train.csv'))
    test_df = pd.read_csv(os.path.join(PATH, 'test.csv'))
    EPOCHS = 25

    print(train_df.head())
    print(test_df.head())
    print(train_df.category_id.value_counts())
    train_df['category_id'] = train_df['category_id'].astype(str)
    IMAGE_HT_WID = 96
    # train_datagen = ImageDataGenerator(
    #     rescale=1. / 255,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     validation_split=0.1)

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.01,
        zoom_range=[0.9, 1.25],
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='reflect',
        # data_format='channels_last',
        brightness_range=[0.5, 1.5],
        validation_split=0.1,
        rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory="dataset/train_images/",
        x_col="file_name",
        y_col="category_id",
        subset="training",
        batch_size=64,
        seed=42,
        shuffle=True,
        class_mode='categorical',
        target_size=(IMAGE_HT_WID, IMAGE_HT_WID))

    valid_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory="dataset/train_images/",
        x_col="file_name",
        y_col="category_id",
        subset="validation",
        batch_size=50,
        seed=42,
        shuffle=True,
        class_mode="categorical",
        target_size=(IMAGE_HT_WID, IMAGE_HT_WID))

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(16, (3, 3), input_shape=(IMAGE_HT_WID, IMAGE_HT_WID, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Conv2D(64, (3, 3)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(tf.keras.layers.MaxPooling2D(2, 2))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(512))
    model.add(tf.keras.layers.Dense(14, activation='softmax'))

    # model.compile(optimizers.rmsprop(lr=0.0001, decay=1e-6),loss="categorical_crossentropy",metrics=["accuracy"])
    model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['acc'])

    model.summary()

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
    history = model.fit_generator(generator=train_generator,
                                  steps_per_epoch=STEP_SIZE_TRAIN,
                                  validation_data=valid_generator,
                                  validation_steps=STEP_SIZE_VALID,
                                  epochs=EPOCHS,
                                  workers=4,
                                  verbose=1)

    model.save('model_save/my_model.h5')
    model.save_weights('model_save/my_model_weights.h5')

    # Evaluation
    # with open('history.json', 'w') as f:
    #     json.dump(history.history, f)

    history_df = pd.DataFrame(history.history)
    # history_df[['loss', 'val_loss']].plot()
    # history_df[['acc', 'val_acc']].plot()

    # Submission
    submission_df = pd.read_csv(os.path.join(PATH, 'sample_submission.csv'))

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_df,
        directory="dataset/test_images/",
        x_col="file_name",
        y_col=None,
        batch_size=50,
        seed=42,
        shuffle=False,
        class_mode=None,
        target_size=(IMAGE_HT_WID, IMAGE_HT_WID))

    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
    test_generator.reset()
    pred = model.predict_generator(test_generator,
                                   steps=STEP_SIZE_TEST,
                                   verbose=1)

    # submission
    predicted_class_indices = np.argmax(pred, axis=1)
    labels = train_generator.class_indices
    labels = dict((v, k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]

    submission = pd.DataFrame({"Id": test_df.id,
                               "Predicted": predictions})
    submission.to_csv("submission.csv", index=False)
