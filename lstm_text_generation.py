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
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.regularizers import l2
import numpy as np
import random
import sys
import json

LSTM_SIZE = 200
EPOCHS = 1
BATCH_SIZE = 500 #128 + 350
LEARNING_RATE = 0.5
DROPOUT = 0.1

TEXT_STEP = 1
# this will be discovered from the text and it longest
# sentence length (Q + A)
MAXLEN = None

# path = sys.argv[1]
PATH = "seinfeld_lstm_corpus.txt"

print('LSTM_SIZE',LSTM_SIZE)
print('EPOCHS',EPOCHS)
print('BATCH_SIZE',BATCH_SIZE)
print('TEXT_STEP',TEXT_STEP)
print('LEARNING_RATE',LEARNING_RATE)
print('MAXLEN',MAXLEN)


def vectorize(text, chars, char_indices, indices_char):
    sentences = []
    next_chars = []
    for i in range(0, len(text) - MAXLEN, TEXT_STEP):
        sentences.append(text[i: i + MAXLEN])
        next_chars.append(text[i + MAXLEN])
    print('nb sequences:', len(sentences))

    print('Vectorization...')
    X = np.zeros((len(sentences), MAXLEN, len(chars)), dtype=np.bool)
    y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
    for i, sentence in enumerate(sentences):
        for t, char in enumerate(sentence):
            X[i, t, char_indices[char]] = 1
        y[i, char_indices[next_chars[i]]] = 1

    print('X', X.shape, 'y', y.shape)
    return X, y


def split_data(text):
    """ Split raw data into chunks: 30% validate, 10% test, 60% train.
    Note that this does *not* split with respect to <q>/<a> or line break.
    """
    X = np.array(list(text), dtype='|S1')
    # this will do an approximate split
    v1, v2, v3, test, tr1, tr2, tr3, tr4, tr5, tr6 = np.array_split(X, 10)
    validate= np.concatenate((v1, v2, v3))
    train= np.concatenate((tr1, tr2, tr3, tr4, tr5, tr6))
    return validate, test, train


def load_corpus(path):
    """ Load and vectorize our corpus, returning arrays that can
    be fed into our LSTM.
    """
    text = open(path).read().lower().replace('\n', '')
    print('corpus length:', len(text))

    # here is where we set global MAXLEN
    global MAXLEN
    MAXLEN = max(map(lambda x: len(x), text.split('<a>')))

    chars = sorted(list(set(text)))
    print('total chars:', len(chars))
    print('chars',chars)

    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    print('char_indices',char_indices)
    print('indices_char',indices_char)

    validate, test, train = split_data(text)
    val_X, val_y = vectorize(validate, chars, char_indices, indices_char)
    ts_X, ts_y = vectorize(test, chars, char_indices, indices_char)
    tr_X, tr_y = vectorize(train, chars, char_indices, indices_char)
    return val_X, val_y, ts_X, ts_y, tr_X, tr_y, chars


def build_model(chars):
    """ Build our model. This can then be used to train or load weights, etc.
    Model is built using the constants at the top of the file.
    """
    # build the model: a single LSTM
    print('Build model...')
    model = Sequential()
    # dropout breaks symmetry on zero-init weights
    model.add(LSTM(
        LSTM_SIZE,
        input_shape=(MAXLEN, len(chars)),
        dropout_W=DROPOUT,
        dropout_U=DROPOUT
    ))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
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


def output_from_seed(model, sentence):
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
            x = np.zeros((1, MAXLEN, len(chars)))
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
    model.save('lstm_window_model.h5')
    with open('lstm_window_char_indices.json','w') as f:
        f.write(json.dumps(char_indices))

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print( "USAGE: {0} path_to_corpus".format(sys.argv[0]))
    #     sys.exit(1)

    val_X, val_y, ts_X, ts_y, tr_X, tr_y, chars = load_corpus(PATH)
    model = build_model(chars)
    train(model, tr_X, tr_y)

    sentence = "you think i'm going down?<q>you're behind in the count.<a>"
    # start_index = random.randint(0, len(text) - maxlen - 1)
    # sentence = text[start_index: start_index + maxlen]

    output_from_seed( model, sentence)
