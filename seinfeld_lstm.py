#!/usr/bin/env python2.7
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation  # , Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras import backend as K
# from keras.utils.data_utils import get_file
# from keras.regularizers import l2
import numpy as np
# import random
import sys
# import json
# import os

# disable verbose logging
K.tf.logging.set_verbosity(K.tf.logging.ERROR)


class SeinfeldAI(object):
    """ Wrapper for building, training and testing a LSTM model
    """
    def __init__(self, lstm_size=200, epochs=1, batch_size=200,
                 learning_rate=0.01, dropout=0.1, activation='softmax',
                 text_step=1, window=40, path='seinfeld_lstm_corpus.txt',
                 debug=True):
        self.LSTM_SIZE = int(lstm_size)
        self.EPOCHS = int(epochs)
        self.BATCH_SIZE = int(batch_size)
        self.LEARNING_RATE = learning_rate
        self.DROPOUT = dropout
        self.ACTIVATION = activation
        self.TEXT_STEP = int(text_step)
        self.WINDOW = int(window)
        self.PATH = path
        if debug:
            logmsg = 'LSTM_SIZE {0} EPOCHS {1} BATCH_SIZE {2} TEXT_STEP {3} ' \
                'LEARNING_RATE {4} WINDOW {5} DROPOUT {6} ACTIVATION {7}'
            print(logmsg.format(
                self.LSTM_SIZE,
                self.EPOCHS,
                self.BATCH_SIZE,
                self.TEXT_STEP,
                self.LEARNING_RATE,
                self.WINDOW,
                self.DROPOUT,
                self.ACTIVATION
            ))

    def vectorize_sentences(self, text, chars, char_indices,
                            indices_char, debug=False):
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
            for i in range(0, len(qa) - self.WINDOW, self.TEXT_STEP):
                sentence = qa[i: i + self.WINDOW]
                sentences.append(sentence)
                next_char = qa[i + self.WINDOW]
                next_chars.append(next_char)
                if debug:
                    print('sentence', sentence, 'next_char', next_char)
            if debug:
                print()
        X = np.zeros((len(sentences), self.WINDOW, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1
        return X, y

    def vectorize(self, text, chars, char_indices, indices_char):
        """ Turn our text into two sets of features, X and y, where
        X is a sliding window of WINDOW characters and y is the next
        character that is coming. Note that the "character" is actually
        a one-hot encoding of the character, so each "character" is represented
        by a vector of zeros and a one indicating the index of the char.
        """
        sentences = []
        next_chars = []
        for i in range(0, len(text) - self.WINDOW, self.TEXT_STEP):
            sentences.append(text[i: i + self.WINDOW])
            next_chars.append(text[i + self.WINDOW])

        X = np.zeros((len(sentences), self.WINDOW, len(chars)), dtype=np.bool)
        y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
        for i, sentence in enumerate(sentences):
            for t, char in enumerate(sentence):
                X[i, t, char_indices[char]] = 1
            y[i, char_indices[next_chars[i]]] = 1

        return X, y

    def split_data(self, text):
        """ Split raw data into chunks: 30% validate, 10% test, 60% train.
        Note that this does *not* split with respect to <q>/<a> or line break.
        """
        X = np.array(list(text), dtype='|S1')
        # this will do an approximate split
        v1, v2, v3, test, tr1, tr2, tr3, tr4, tr5, tr6 = np.array_split(X, 10)
        validate = np.concatenate((v1, v2, v3))
        train = np.concatenate((tr1, tr2, tr3, tr4, tr5, tr6))
        return validate, test, train

    def load_corpus(self, path):
        """ Load and vectorize our corpus, returning arrays that can
        be fed into our LSTM.
        """
        text = open(path).read().lower().replace('\n', '')

        chars = sorted(list(set(text)))

        char_indices = dict((c, i) for i, c in enumerate(chars))
        indices_char = dict((i, c) for i, c in enumerate(chars))
        # print('char_indices', char_indices)
        # print('indices_char', indices_char)

        validate, test, train = self.split_data(text)
        val_X, val_y = self.vectorize_sentences(
            validate, chars, char_indices, indices_char
        )
        ts_X, ts_y = self.vectorize_sentences(
            test, chars, char_indices, indices_char)
        tr_X, tr_y = self.vectorize_sentences(
            train, chars, char_indices, indices_char)
        return val_X, val_y, ts_X, ts_y, tr_X, tr_y, chars, \
            char_indices, indices_char

    def build_model(self, chars):
        """ Build our model. This can then be used to train or load weights, etc.
        Model is built using the constants at the top of the file.
        """
        # build the model: a single LSTM
        # print('Build model...')
        model = Sequential()
        # dropout breaks symmetry on zero-init weights
        model.add(LSTM(
            self.LSTM_SIZE,
            input_shape=(self.WINDOW, len(chars)),
            dropout_W=self.DROPOUT,
            dropout_U=self.DROPOUT
        ))
        model.add(Dense(len(chars)))
        # other options include relu
        model.add(Activation(self.ACTIVATION))
        optimizer = RMSprop(lr=self.LEARNING_RATE)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    def train(self, model, X, y):
        """ Train our model for epochs, returning accuracy history
        """
        return model.fit(
            X, y,
            batch_size=self.BATCH_SIZE,
            nb_epoch=self.EPOCHS)

    def sample(self, preds, temperature=1.0):
        """ Return softmax with "temperature" scores.
        """
        # helper function to sample an index from a probability array
        preds = np.asarray(preds).astype('float64')
        preds = np.log(preds) / temperature
        exp_preds = np.exp(preds)
        preds = exp_preds / np.sum(exp_preds)
        probas = np.random.multinomial(1, preds, 1)
        return np.argmax(probas)

    def output_from_seed(self, model, sentence, char_indices,
                         indices_char, chars):
        """ Take a seed sentence and generate some output from out LSTM
        """
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
                x = np.zeros((1, self.WINDOW, len(chars)))
                for t, char in enumerate(sentence):
                    x[0, t, char_indices[char]] = 1.
                preds = model.predict(x)[0]
                next_index = self.sample(preds, diversity)
                next_char = indices_char[next_index]
                generated += next_char
                sentence = sentence[1:] + next_char
                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

    def save_model(self, model):
        """ Write model to disk
        """
        name = 'lstm_model_{0}_{1}_{2}_{3}_{4}_{5}_{6}_{7}_h5'.format(
            self.LSTM_SIZE,
            self.EPOCHS,
            self.BATCH_SIZE,
            self.LEARNING_RATE,
            self.DROPOUT,
            self.ACTIVATION,
            self.TEXT_STEP,
            self.WINDOW)
        model.save(name)

    def test_model(self, model, X, y):
        """ Take a set of inputs and test our trained LSTM, returning loss
        """
        score = model.evaluate(
            X, y,
            batch_size=self.BATCH_SIZE)
        return score

    def run(self):
        """ Generate, train and test a model, returning train, test loss
        """
        val_X, val_y, ts_X, ts_y, tr_X, tr_y, chars, \
            char_indices, indices_char = self.load_corpus(self.PATH)
        model = self.build_model(chars)
        history = self.train(model, tr_X, tr_y)
        training_loss = history.history['loss'][-1]
        # print('# RESULTS')
        # print("~ train_score", training_loss)
        # # sentence = "hey. where's the car?<q>"
        # # # start_index = random.randint(0, len(text) - maxlen - 1)
        # # # sentence = text[start_index: start_index + maxlen]
        # # output_from_seed(model, sentence, char_indices, indices_char, chars)
        test_loss = self.test_model(model, ts_X, ts_y)
        # print('~ test_score', test_loss)
        return training_loss, test_loss


def five_models(**kwargs):
    """ Build, train and test five models with the specified params. Return
    averaged loss for train and test between all five iterations. Passes
    all kwargs (hyperparams) onto SeinfeldAI class initialization.
    """
    lstm_model = SeinfeldAI(**kwargs)
    tr_err = 0
    ts_err = 0
    for i in range(5):
        training_loss, test_loss = lstm_model.run()
        tr_err += training_loss
        ts_err += test_loss
    return tr_err/5, ts_err/5


if __name__ == "__main__":
    five_models()
