import pandas as pd
import os

PATH = PATH = 'dataset/'


def pre_processing_image():
    train_df = pd.read_csv(os.path.join(PATH, 'train.csv'))
    test_df = pd.read_csv(os.path.join(PATH, 'test.csv'))
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
    pre_processing_image()
