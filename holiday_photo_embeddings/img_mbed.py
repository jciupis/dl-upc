import sys, time
import numpy as np
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import os


EMBEDDING_SIZE = 4096
BATCH_SIZE = 5
IMG_SIZE = 224
IMG_PATH = 'E:/slovenia/all_days_resized'


def input_pipeline(img_paths, batch_size):
    """
    Create a batch generator.

    :param img_paths: list, strings containing paths to image files
    :param batch_size: int, single batch size
    :return: generator of batches
    """
    # Iterate over all image paths
    for n in range(len(img_paths)):
        x_batch = np.zeros((0, IMG_SIZE, IMG_SIZE, 3))
        y_batch = []
        for i in range(batch_size):
            try:
                img_path = img_paths.pop(0)
            except IndexError:
                break

            img = image.load_img(img_path, target_size=(224, 224))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x_batch = np.concatenate((x_batch, x), axis=0)
            y_batch.append(img_path)

        if x_batch.shape[0] != 0:
            yield x_batch, y_batch


def generate_embeddings():
    """
    Generate image embeddings and save them to a .npz file.
    """
    # Load the VGG16 architecture with weights pre-trained using ImageNet2012
    base_model = VGG16(weights='imagenet')

    # Define a model that for given input outputs activations from the layer one before last.
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

    # Create variables for embeddings generation.
    embeddings = np.zeros((0, EMBEDDING_SIZE))
    labels = []
    images = [os.path.join(IMG_PATH, f) for f in os.listdir(IMG_PATH)]
    step = 0

    nb_steps = int(len(images) / BATCH_SIZE)
    if len(images) % BATCH_SIZE > 0:
        nb_steps += 1

    # Generate embeddings.
    for x_batch, y_batch in input_pipeline(images, BATCH_SIZE):
        x = preprocess_input(x_batch)
        batch_embeddings = model.predict(x)
        embeddings = np.concatenate((embeddings, batch_embeddings), axis=0)
        labels.extend(y_batch)

        print('Progress: ' + str((step / nb_steps) * 100) + '%')
        step += 1

    # Save embeddings to a file.
    print('Saving dataset embeddings into img_mbed.npz file...')
    np.savez('img_mbed.npz', embeddings=embeddings, labels=labels)


def main():
    generate_embeddings()


if __name__ == '__main__':
    main()