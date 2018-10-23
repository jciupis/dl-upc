# Dog breed classification using neural networks

This is an academic project using various kinds of neural networks to classify dog breeds based on images from the
[Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) dataset.

## Getting started
To run the code, first download the [Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) dataset. The code
has been written for Python 3 using Keras, matplotlib and PIL libraries, so make sure to download the necessary dependencies.
Replace the paths to images and their annotations in util.py with the appropriate paths on your machine. Run util.py to
generate resized images and create encodings and annotations .json files.

With the resized images and .json files generated, fnn.py and cnn.py can be run to train feedforward and convolutional neural network respectively.

## Results
The best top-1 accuracies achieved with this code are 8.553% for fnn.py and 35.992% for cnn.py. 
