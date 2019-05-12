import os
import json
import pandas as pd
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.layers import LeakyReLU
import os


def pre_processing_image():
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')
    print(train_df.head())
    print(test_df.head())
    print("Train and test shape: {} {}".format(train_df.shape, test_df.shape))

    # Data exploration
    # Let's define the classes:
    classes_wild = {0: 'empty', 1: 'deer', 2: 'moose', 3: 'squirrel', 4: 'rodent', 5: 'small_mammal', \
                    6: 'elk', 7: 'pronghorn_antelope', 8: 'rabbit', 9: 'bighorn_sheep', 10: 'fox', 11: 'coyote', \
                    12: 'black_bear', 13: 'raccoon', 14: 'skunk', 15: 'wolf', 16: 'bobcat', 17: 'cat', \
                    18: 'dog', 19: 'opossum', 20: 'bison', 21: 'mountain_goat', 22: 'mountain_lion'}

    train_df['classes_wild'] = train_df['category_id'].apply(lambda cw: classes_wild[cw])
    print(classes_wild)

    ## Check images
    train_image_files = list('dataset/train_images')
    test_image_files = list('dataset/test_images')

    print("Number of image files: train:{} test:{}".format(len(train_image_files), len(test_image_files)))

    train_file_names = list(train_df['file_name'])
    print("Matching train image names: {}".format(len(set(train_file_names).intersection(train_image_files))))

    test_file_names = list(test_df['file_name'])
    print("Matching test image names: {}".format(len(set(test_file_names).intersection(test_image_files))))


if __name__ == '__main__':
    train_df = pd.read_csv('dataset/train.csv')
    test_df = pd.read_csv('dataset/test.csv')

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

    test_datagen = ImageDataGenerator(rescale=1. / 255)
    # test_datagen = ImageDataGenerator()

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

    EPOCHS = 1
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
    history_df = pd.DataFrame(history.history)

    # Submission
    y_test = model.predict(test_df)

    submission_df = pd.read_csv('dataset/sample_submission.csv')
    submission_df['Predicted'] = y_test.argmax(axis=1)
    print(submission_df.shape)
    submission_df.head()

    submission_df.to_csv('submission.csv', index=False)
    history_df.to_csv('history.csv', index=False)
    with open('history.json', 'w') as f:
        json.dump(history.history, f)
