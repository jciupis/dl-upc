from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
import numpy as np
import random
import time
import os


CPU_IMPL = 0
GPU_IMPL = 2


def sample(predictions, temperature=1.0):
    """
    Sample an index from a probability array to balance distribution obtained by the model.

    :param predictions: model's predictions
    :param temperature: temperature of prediction
    :return: next predicted character
    """
    predictions = np.asarray(predictions).astype('float64')
    predictions = np.log(predictions) / temperature
    exp_predictions = np.exp(predictions)
    predictions = exp_predictions / np.sum(exp_predictions)
    probabilities = np.random.multinomial(1, predictions, 1)
    return np.argmax(probabilities)


def generate_text(seed, numlines, gen_file, wseed=False):
    """
    Generates a number of lines (or at most 1000 characters) using the given seed
    :param seed:
    :param lines:
    :return:
    """
    generated = ''
    gprinted = ''
    sentence = seed
    generated += sentence

    nlines = 0
    for i in range(1000):
        x = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(sentence):
            x[0, t, char_indices[char]] = 1.

        predictions = model.predict(x, verbose=0)[0]
        next_index = sample(predictions, diversity)
        next_char = indices_char[next_index]
        generated += next_char
        gprinted += next_char

        sentence = sentence[1:] + next_char
        # Count the number of lines generated
        if next_char == '\n':
            nlines += 1
        if nlines > numlines:
            break

    if wseed:
        gen_file.write(seed + gprinted)
    else:
        gen_file.write(gprinted)

    gen_file.write('\n')
    gen_file.flush()


def random_seed(chars, nb_chars):
    """
    Generate a random string.

    :param chars: list of characters used to generate the string
    :param nb_chars: length of the generated string
    :return: random string
    """
    s = ''
    for i in range(nb_chars):
        s += chars[random.randint(0, len(chars) - 1)]

    return s


if __name__ == '__main__':
    print("Starting: ", time.ctime())

    ########################################################################################
    # Data

    file = 'India_Pale_Ale_(IPA).txt'  # TODO: parse the filename as cmd argument
    with open(file, 'r') as f:
        text = f.read().lower()
    print('Total number of characters in the corpus: ', len(text))

    chars = sorted(list(set(text)))
    print('Total number of unique characters in the corpus: ', len(chars))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))

    # cut the text in semi-redundant sequences of maxlen characters
    maxlen = 50
    step = 3
    sentences = []
    next_chars = []
    for i in range(0, len(text) - maxlen, step):
        sentences.append(text[i: i + maxlen])
        next_chars.append(text[i + maxlen])
    print('Total number of sequences: ', len(sentences))

    # Vectorizes the sequences with one hot encoding
    print('Vectorization...')
    X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    ########################################################################################
    # Model
    print('Build model...')

    RNN = LSTM  # GRU
    IMPL = GPU_IMPL  # CPU_IMPL
    lsize = 256
    nlayers = 3
    dropout = 0

    model = Sequential()
    if nlayers == 1:
        model.add(RNN(lsize, input_shape=(maxlen, len(chars)), implementation=IMPL, recurrent_dropout=dropout))
    else:
        model.add(RNN(lsize, input_shape=(maxlen, len(chars)), implementation=IMPL, recurrent_dropout=dropout, return_sequences=True))
        for i in range(1, nlayers - 1):
            model.add(RNN(lsize, implementation=IMPL, recurrent_dropout=dropout, return_sequences=True))
        model.add(RNN(lsize, implementation=IMPL, recurrent_dropout=dropout))
    model.add(Dense(len(chars), activation='softmax'))

    ########################################################################################
    # Training
    model.compile(loss='categorical_crossentropy', optimizer="adam")

    batch_size = 512
    iterations = 100
    nb_epochs = 10

    # File for saving the generated text each iteration
    filename = os.path.splitext(os.path.basename(file))[0]
    gen_file = open('generated-TXT%s-ML%d-S%d-NL%d-D%3.2f-BS%d.txt' % (filename, maxlen, lsize, nlayers, dropout, batch_size), 'w')

    # train the model, output generated text after each iteration
    for iteration in range(iterations):
        print()
        print('-' * 50)
        print('Iteration %d' %(iteration + 1))
        model.fit(X, y, batch_size=batch_size, epochs=nb_epochs)

        gen_file.write('-' * 50 + '\n')
        gen_file.write(time.ctime() + '\n')
        gen_file.write('Iteration %d\n' % (iteration + 1))
        seed = random_seed(chars, maxlen)
        for diversity in [0.2, 0.4, 0.8, 1.0]:
            gen_file.write('\n\n')
            gen_file.write('DIV = %3.2f\n\n' % diversity)
            generate_text(seed, numlines=10, gen_file=gen_file, wseed=False)
    gen_file.close()
    print()
    print("Ending:", time.ctime())
