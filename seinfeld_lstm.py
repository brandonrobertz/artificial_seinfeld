#!/usr/bin/env python2.7
from __future__ import print_function
from keras.models import Sequential  # , load_model
from keras.layers import Dense, Activation  # , Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras import backend as K
# from keras.utils.data_utils import get_file
# from keras.regularizers import l2
import numpy as np
# import random
# import sys
from functools import reduce
import cPickle as pickle
# import os
import time

# disable verbose logging
K.tf.logging.set_verbosity(K.tf.logging.ERROR)


class SeinfeldAI(object):
    """ Wrapper for building, training and testing a LSTM model
    """

    def __init__(self, lstm_size=200, epochs=1, batch_size=128,
                 learning_rate=0.01, dropout=0.1, activation='softmax',
                 text_step=1, window=40, path='seinfeld_lstm_corpus.jerry.txt',
                 debug=True, character='jerry', write_model=True):
        self.lstm_size = int(lstm_size)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.activation = activation
        self.text_step = int(text_step)
        self.window = int(window)
        self.path = path
        self.character = character
        self.write_model = write_model
        # these need to be hydrated via vectorization or load_model
        self.chars = None
        self.char_indices = None
        self.indices_char = None
        if debug:
            logmsg = 'lstm_size {0} epochs {1} batch_size {2} text_step {3} ' \
                'learning_rate {4} window {5} dropout {6} activation {7} ' \
                'path {8} CHARACTER {9}'
            print(logmsg.format(
                self.lstm_size,
                self.epochs,
                self.batch_size,
                self.text_step,
                self.learning_rate,
                self.window,
                self.dropout,
                self.activation,
                self.path,
                self.character
            ))

    def vectorize_sentences(self, text, debug=False):
        """ Strategy: instead of sliding a window across an entire
        corpus as one long string, we should slide a window forward
        starting with each sentence. when we hit the end of a response,
        we move the window to the start of the next question, filling
        the entire window.
        """
        joined = reduce(str.__add__, text)
        # <a> always terminates a question/answer series
        qas = map(lambda x: x + '<a>', joined.split('<a>'))
        # TODO: shuffle qas
        window_chunks = []
        next_chars = []
        for qa in qas:
            for i in range(0, len(qa) - self.window, self.text_step):
                sentence = qa[i: i + self.window]
                window_chunks.append(sentence)
                next_char = qa[i + self.window]
                next_chars.append(next_char)
                if debug:
                    print('sentence', sentence, 'next_char', next_char)
            if debug:
                print()
        X_shape = (len(window_chunks), self.window, len(self.chars))
        X = np.zeros(X_shape, dtype=np.bool)
        y_shape = (len(window_chunks), len(self.chars))
        y = np.zeros(y_shape, dtype=np.bool)
        for i, sentence in enumerate(window_chunks):
            for t, char in enumerate(sentence):
                X[i, t, self.char_indices[char]] = 1
            y[i, self.char_indices[next_chars[i]]] = 1
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

        self.chars = sorted(list(set(text)))

        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))
        # print('char_indices', char_indices)
        # print('indices_char', indices_char)

        validate, test, train = self.split_data(text)
        val_X, val_y = self.vectorize_sentences(validate)
        ts_X, ts_y = self.vectorize_sentences(test)
        tr_X, tr_y = self.vectorize_sentences(train)
        return val_X, val_y, ts_X, ts_y, tr_X, tr_y

    def build_model(self):
        """ Build our model. This can then be used to train or load weights, etc.
        Model is built using the constants at the top of the file.
        """
        # build the model: a single LSTM
        # print('Build model...')
        model = Sequential()
        # dropout breaks symmetry on zero-init weights
        model.add(LSTM(
            self.lstm_size,
            input_shape=(self.window, len(self.chars)),
            init='zero',
            dropout_W=self.dropout,
            dropout_U=self.dropout,
            unroll=True,
            consume_less='mem'
        ))
        model.add(Dense(len(self.chars)))
        # other options include relu
        model.add(Activation(self.activation))
        optimizer = RMSprop(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer)
        return model

    def train(self, model, X, y):
        """ Train our model for epochs, returning accuracy history
        """
        return model.fit(
            X, y,
            batch_size=self.batch_size,
            nb_epoch=self.epochs)

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

    def output_from_seed(self, model, sentence):
        """ Take a seed sentence and generate some output from out LSTM
        """
        sentence = sentence.lower()
        if sentence[-3:] != '<q>':
            sentence += '<q>'

        # this will be the entire sentence vectorized, since we're not training
        # we ignore the targets
        X, y = self.vectorize_sentences(sentence)

        # seed the network before we attempt to collect output
        model.predict(X)
        # print('Last supervised pred',
        #       self.indices_char[self.sample(y[-1], 0.2)])

        # this seeds the > in <q>
        X, y = self.vectorize_sentences(sentence[-self.window:] + '>')
        p = model.predict(X)
        pred_char = self.indices_char[self.sample(p[0], 0.2)]

        print('----- Generating with seed: "' + sentence + '"')

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('-- diversity:', diversity)

            generated = ''
            for i in range(60):
                blank = np.zeros((1, self.window, len(self.chars)))
                blank[0][-1][self.char_indices[pred_char]] = 1.0
                X = np.vstack((X[1:], blank))
                preds = model.predict(X)[0]
                pred_ix = self.sample(preds, diversity)
                pred_char = self.indices_char[pred_ix]
                generated += pred_char

            print(generated)

    def save_model(self, model):
        """ Write model to disk
        """
        ts = int(time.time())
        name_pfx = 'models/model_{0}'.format(ts)
        modelname = name_pfx + '.h5'
        model.save(modelname)
        auxname = name_pfx + '.aux.p'
        with open(auxname, 'w') as f:
            pickle.dump({
                'chars': self.chars,
                'char_indices': self.char_indices,
                'indices_char': self.indices_char,
                'model_params': {
                    "lstm_size": self.lstm_size,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "learning_rate": self.learning_rate,
                    "dropout": self.dropout,
                    "activation": self.activation,
                    "text_step": self.text_step,
                    "window": self.window,
                    "path": self.path,
                    "character": self.character,
                    "write_model": self.write_model
                }
            }, f)
            print('Saved model to', modelname, '&', auxname)

    def load_model(self, model_h5_path):
        """ Build a model and load weights from disk
        """
        auxfile = model_h5_path.replace('.h5', '') + '.aux.p'
        with open(auxfile, 'r') as f:
            aux = pickle.load(f)
            self.chars = aux['chars']
            self.char_indices = aux['char_indices']
            self.indices_char = aux['indices_char']
            self.lstm_size = aux["model_params"]["lstm_size"]
            self.epochs = aux["model_params"]["epochs"]
            self.batch_size = aux["model_params"]["batch_size"]
            self.learning_rate = aux["model_params"]["learning_rate"]
            self.dropout = aux["model_params"]["dropout"]
            self.activation = aux["model_params"]["activation"]
            self.text_step = aux["model_params"]["text_step"]
            self.window = aux["model_params"]["window"]
            self.path = aux["model_params"]["path"]
            self.character = aux["model_params"]["character"]
            self.write_model = aux["model_params"]["write_model"]
        model = self.build_model()
        model.load_weights(model_h5_path)
        # model = load_model(model_path)
        return model

    def test_model(self, model, X, y):
        """ Take a set of inputs and test our trained LSTM, returning loss
        """
        p = model.predict(X)
        if np.isnan(p).any():
            return 100
        score = model.evaluate(
            X, y,
            batch_size=self.batch_size)
        self.output_from_seed(model, 'hey jerry<q>')
        return score

    def run(self):
        """ Generate, train and test a model, returning train, test loss
        """
        val_X, val_y, ts_X, ts_y, tr_X, tr_y, = self.load_corpus(self.path)
        model = self.build_model()
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
        if self.write_model:
            self.save_model(model)
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
    # # this model returns nans for output
    # ai = SeinfeldAI(lstm_size=302,epochs=2,batch_size=128,text_step=1,
    #                 learning_rate=0.110421159239,window=188,
    #                 dropout=0.892325476646,
    #                 path='seinfeld_lstm_corpus.jerry.txt')
    # ai.run()
