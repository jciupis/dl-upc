import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
import util

def plot_accuracy(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('model_accuracy.pdf')
    plt.close()


def plot_loss(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('model_loss.pdf')


def clf_statistics(nn, x_test, y_test):
    # Compute probabilities and assign most probable label
    y_pred = np.argmax(nn.predict(x_test), axis=1)

    # Save statistics
    with open('one_hot_encodings.json', 'r') as file:
        one_hot_encodings = json.load(file)

    one_hot_to_breed = util.one_hot_encoding_to_class(one_hot_encodings)
    breeds = one_hot_to_breed.values()

    clf_report = classification_report(np.argmax(y_test, axis=1), y_pred, target_names=breeds)
    cnf_matrix = confusion_matrix(np.argmax(y_test, axis=1), y_pred)

    opt = np.get_printoptions()
    np.set_printoptions(threshold=np.inf)

    with open('clf_report.txt', 'w') as file:
        print(clf_report, file=file)

    with open('confusion_matrix.txt', 'w') as file:
        print(cnf_matrix, file=file)

    np.set_printoptions(**opt)
