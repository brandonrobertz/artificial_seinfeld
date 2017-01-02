#!/usr/bin/env python2.7
'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation  # , Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
# from keras.utils.data_utils import get_file
# from keras.regularizers import l2
import numpy as np
# import random
import sys
# import json
import os

LSTM_SIZE = int(os.environ.get('LSTM_SIZE', 200))
EPOCHS = int(os.environ.get('LSTM_EPOCHS', 1))
BATCH_SIZE = int(os.environ.get('LSTM_BATCH_SIZE', 200))
LEARNING_RATE = float(os.environ.get('LSTM_LEARNING_RATE', 0.01))
DROPOUT = float(os.environ.get('LSTM_DROPOUT', 0.1))
ACTIVATION = os.environ.get('LSTM_ACTIVATION', 'softmax')

TEXT_STEP = int(os.environ.get('LSTM_TEXT_STEP', 10))
# this will be discovered from the text and it longest
# sentence length (Q + A)
WINDOW = int(os.environ.get('LSTM_WINDOW', 40))

# path = sys.argv[1]
PATH = "seinfeld_lstm_corpus.txt"


print('# PARAMS')
print('~ LSTM_SIZE', LSTM_SIZE)
print('~ EPOCHS', EPOCHS)
print('~ BATCH_SIZE', BATCH_SIZE)
print('~ TEXT_STEP', TEXT_STEP)
print('~ LEARNING_RATE', LEARNING_RATE)
print('~ WINDOW', WINDOW)
print('~ ACTIVATION', ACTIVATION)


def vectorize_sentences(text, chars, char_indices, indices_char, dbg=False):
    """ Strategy: instead of sliding a window across an entire
    corpus as one long string, we should slide a window forward
    starting with each sentence. when we hit the end of a response,
    we move the window to the start of the next question, filling
    the entire window.
    """
    joined = reduce(str.__add__, text)
    # <a> always terminates a question/answer series
    qas = map(lambda x: x + '<a>', joined.split('<a>'))
    sentences = []
    next_chars = []
    for qa in qas:
        for i in range(0, len(qa) - WINDOW, TEXT_STEP):
            sentence = qa[i: i + WINDOW]
            sentences.append(sentence)
            next_char = qa[i + WINDOW]
            next_chars.append(next_char)
            if dbg:
                print('sentence', sentence, 'next_char', next_char)
        if dbg:
            print()
    X = np.zeros((len(sentences), WINDOW, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1
    return X, y


def vectorize(text, chars, char_indices, indices_char):
    """ Turn our text into two sets of features, X and y, where
    X is a sliding window of WINDOW characters and y is the next
    character that is coming. Note that the "character" is actually
    a one-hot encoding of the character, so each "character" is represented
    by a vector of zeros and a one indicating the index of the char.
    """
    sentences = []
    next_chars = []
    for i in range(0, len(text) - WINDOW, TEXT_STEP):
        sentences.append(text[i: i + WINDOW])
        next_chars.append(text[i + WINDOW])

    X = np.zeros((len(sentences), WINDOW, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    return X, y


def split_data(text):
    """ Split raw data into chunks: 30% validate, 10% test, 60% train.
    Note that this does *not* split with respect to <q>/<a> or line break.
    """
    X = np.array(list(text), dtype='|S1')
    # this will do an approximate split
    v1, v2, v3, test, tr1, tr2, tr3, tr4, tr5, tr6 = np.array_split(X, 10)
    validate = np.concatenate((v1, v2, v3))
    train = np.concatenate((tr1, tr2, tr3, tr4, tr5, tr6))
    return validate, test, train


def load_corpus(path):
    """ Load and vectorize our corpus, returning arrays that can
    be fed into our LSTM.
    """
    text = open(path).read().lower().replace('\n', '')

    # # here is where we set global WINDOW
    # global WINDOW
    # WINDOW = max(map(lambda x: len(x), text.split('<a>')))

    chars = sorted(list(set(text)))

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    # print('char_indices', char_indices)
    # print('indices_char', indices_char)

    validate, test, train = split_data(text)
    val_X, val_y = vectorize_sentences(
        validate, chars, char_indices, indices_char
    )
    ts_X, ts_y = vectorize_sentences(test, chars, char_indices, indices_char)
    tr_X, tr_y = vectorize_sentences(train, chars, char_indices, indices_char)
    return val_X, val_y, ts_X, ts_y, tr_X, tr_y, chars, \
        char_indices, indices_char


def build_model(chars):
    """ Build our model. This can then be used to train or load weights, etc.
    Model is built using the constants at the top of the file.
    """
    # build the model: a single LSTM
    # print('Build model...')
    model = Sequential()
    # dropout breaks symmetry on zero-init weights
    model.add(LSTM(
        LSTM_SIZE,
        input_shape=(WINDOW, len(chars)),
        dropout_W=DROPOUT,
        dropout_U=DROPOUT
    ))
    model.add(Dense(len(chars)))
    # other options include relu
    model.add(Activation(ACTIVATION))
    optimizer = RMSprop(lr=LEARNING_RATE)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model


def train(model, X, y):
    return model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def output_from_seed(model, sentence, char_indices, indices_char, chars):
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        # let's only show the generated output, not mixed
        # generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        print()
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, WINDOW, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.
            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            generated += next_char
            sentence = sentence[1:] + next_char
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()


def save_model(model):
    model.save(
        'lstm_window_model_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_h5'.format(
            LSTM_SIZE,
            EPOCHS,
            BATCH_SIZE,
            LEARNING_RATE,
            DROPOUT,
            ACTIVATION,
            TEXT_STEP,
            WINDOW))


def test_model(model, X, y):
    score = model.evaluate(X, y, batch_size=BATCH_SIZE)
    # print('Test score:', score)
    # print('Test accuracy:', acc)
    return score


def epoch():
    val_X, val_y, ts_X, ts_y, tr_X, tr_y, chars, \
        char_indices, indices_char = load_corpus(PATH)
    model = build_model(chars)
    history = train(model, tr_X, tr_y)
    training_loss = history.history['loss'][-1]
    # print('# RESULTS')
    # print("~ train_score", training_loss)
    # # sentence = "hey. where's the car?<q>"
    # # # start_index = random.randint(0, len(text) - maxlen - 1)
    # # # sentence = text[start_index: start_index + maxlen]
    # # output_from_seed(model, sentence, char_indices, indices_char, chars)
    test_loss = test_model(model, ts_X, ts_y)
    # print('~ test_score', test_loss)
    return training_loss, test_loss


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print( "USAGE: {0} path_to_corpus".format(sys.argv[0]))
    #     sys.exit(1)

    tr_err = 0
    ts_err = 0
    for i in range(5):
        training_loss, test_loss = epoch()
        tr_err += training_loss
        ts_err += test_loss
    print('* avg train err', tr_err/5)
    print('* avg test err', ts_err/5)
