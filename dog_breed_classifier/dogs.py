import json
import numpy as np
import util

class Dogs:
    """
    The Dog class represents the Stanford dogs dataset in a neural network friendly way.
    """
    def __init__(self, image_width=64, image_height=64):
        """
        The constructor loads the one hot encodings and image annotations..
        :param image_width:
        :param image_height:
        """
        self.image_height = image_height
        self.image_width = image_width

        with open('one_hot_encodings.json', 'r') as one_hot_encodings:
            self.one_hot_encodings = json.load(one_hot_encodings)

        with open('train_annotations.json', 'r') as train_annotations:
            self.train_annotations = json.load(train_annotations)

        with open('test_annotations.json', 'r') as test_annotations:
            self.test_annotations = json.load(test_annotations)

    def __load_training_data(self):
        """
        This function loads training data into RAM.

        :return:
        x_train: float16 numpy array of training RGB image data with shape
                (num_samples, self.image_width, self.image_height, 3)
        y_train: numpy array of training one hot encodings of dog breed of corresponding x_train images
        """
        train_samples = []

        for i, breed in enumerate(self.train_annotations.keys()):
            for j, annotation in enumerate(self.train_annotations[breed]):
                sample_x = util.get_resized_image_data(annotation, self.image_width, self.image_height)
                sample_y = self.one_hot_encodings[breed]
                train_samples.append((sample_x, sample_y))

        np.random.shuffle(train_samples)
        x_train, y_train = map(list, zip(*train_samples))

        return np.array(x_train), np.array(y_train)

    def __load_test_data(self):
        """
        This function loads test data into RAM.

        :return:
        x_test: float16 numpy array of test RGB image data with shape
                (num_samples, self.image_width, self.image_height, 3)
        y_test: numpy array of test one hot encodings of dog breed of corresponding x_test images
        """
        x_test = []
        y_test = []

        for i, breed in enumerate(self.test_annotations.keys()):
            for j, annotation in enumerate(self.test_annotations[breed]):
                x_test.append(util.get_resized_image_data(annotation, self.image_width, self.image_height))
                y_test.append(self.one_hot_encodings[breed])

        return np.array(x_test), np.array(y_test)

    def load_data(self):
        """
        This function loads the dataset into RAM
        :return:
        x_train: float16 numpy array of training RGB image data with shape
                (num_samples, self.image_width, self.image_height, 3)
        y_train: numpy array of training one hot encodings of dog breed of corresponding x_train images
        x_test: float16 numpy array of test RGB image data with shape
                (num_samples, self.image_width, self.image_height, 3)
        y_test: numpy array of test one hot encodings of dog breed of corresponding x_test images
        """

        print('Loading training set...')
        x_train, y_train = self.__load_training_data()

        print('Loading test set...')
        x_test, y_test = self.__load_test_data()

        return (x_train, y_train), (x_test, y_test)


def main():
    (x_train, y_train), (x_test, y_test) = Dogs().load_data()
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


if __name__ == '__main__':
    main()
