#!/usr/bin/env python2.7
from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras import backend as K
import numpy as np
from functools import reduce
import cPickle as pickle
import time
import settings

# disable verbose logging
K.tf.logging.set_verbosity(K.tf.logging.ERROR)


class SeinfeldAI(object):
    """ Wrapper for building, training and testing a LSTM model
    """

    def __init__(self, lstm_size=200, epochs=1, batch_size=128,
                 learning_rate=0.01, dropout_W=0.1, dropout_U=0.1,
                 activation='softmax',
                 text_step=1, window=40, path='seinfeld_lstm_corpus.jerry.txt',
                 debug=True, character='jerry', write_model=True):
        self.lstm_size = int(lstm_size)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = learning_rate
        self.dropout_W = dropout_W
        self.dropout_U = dropout_U
        self.activation = activation
        self.text_step = int(text_step)
        self.window = int(window)
        self.path = path
        self.character = character
        self.write_model = write_model
        # this needs to be loaded or built
        self.model = None
        # these need to be hydrated via vectorization or load_model
        self.chars = None
        self.char_indices = None
        self.indices_char = None
        # these cannot be present in the corpus
        self.end_q_seq = settings.END_Q_SEQ  # '|'
        self.end_a_seq = settings.END_A_SEQ  # '#'
        if debug:
            logmsg = 'lstm_size {0} epochs {1} batch_size {2} text_step {3} ' \
                'learning_rate {4} window {5} dropout_W {6} U {7} ' \
                'activation {8} ' \
                'path {9} CHARACTER {10}'
            print(logmsg.format(
                self.lstm_size,
                self.epochs,
                self.batch_size,
                self.text_step,
                self.learning_rate,
                self.window,
                self.dropout_W,
                self.dropout_U,
                self.activation,
                self.path,
                self.character
            ))

    def vectorize_sentences(self, text, debug=False, has_answer=True):
        """ Strategy: instead of sliding a window across an entire
        corpus as one long string, we should slide a window forward
        starting with each sentence. when we hit the end of a response,
        we move the window to the start of the next question, filling
        the entire window.
        PARAMS:
            answer: (default, True) indicated whether the text contains
              answers (training examples) in addition to questions
        """
        # <a> always terminates a question/answer series, if we have a q/a pair
        # then we want to split the raw text corpus into chunks that contain
        # both the <q> and the <a>, otherwise, just split on <q>
        split_seq = self.end_a_seq
        if not has_answer:
            assert(text.count(self.end_q_seq) <= 1)
            split_seq = self.end_q_seq

        # split can leave a blank ste at end of array, filter them out
        full_sentences = filter(lambda x: x, text.split(split_seq))
        # we need to restore all end sequences, stripped via split
        qas = map(lambda x: x + split_seq, full_sentences)

        if debug:
            print('qas', qas)

        # TODO: shuffle qas or maybe do this higher up in the stack (split_data)?
        window_chunks = []
        next_chars = []
        for qa in qas:
            if debug:
                print('qa', qa)

            # step 1: check if question length is longer than window
            question, answer = qa.split(self.end_q_seq)
            question += self.end_q_seq

            # step 1a: question > window: make chunks
            #   the size of the window, starting at beginning, y being the next char,
            #   until we get to the end of the question delimeter
            if len(question) > self.window:
                if debug:
                    print('len(question) is > self.window')
                for i in range(0, len(question) - self.window, self.text_step):
                    window_chunk = question[i: i + self.window]
                    window_chunks.append(window_chunk)
                    next_char = question[i + self.window]
                    next_chars.append(next_char)
                    if debug:
                        print(i, 'x', window_chunk, 'y', next_char)

            # step 1b: question <= window: add a single chunk and continue, y will be
            #   end of the question delim (>, currently)
            else:
                if debug:
                    print('len(question> is <= self.window')
                window_chunk = question[:-1]
                window_chunks.append(window_chunk)
                next_char = question[-1]
                next_chars.append(next_char)

            # step 2: if we're in answer=False mode, we're done with this qa pair
            if not has_answer:
                # this facilitates seeding for question answering directly.
                # the strategy with this will be to seed on the entire returned
                # set of vectors and, at the end of the array, the network is seeded
                # (since has_answer=True only supports a single-question)
                window_chunk = question[-1 * self.window:]
                window_chunks.append(window_chunk)
                if debug:
                    print('No answer, continuing!')
                continue

            # step 3: now we need to continue to slide the window forward, starting
            #   with y = first char of answer until end of answer
            start = len(question) - self.window
            end = len(qa) - self.window
            if debug:
                print('start', start, 'end', end)
            for i in range(start, end, self.text_step):
                window_chunk = qa[i: i + self.window]
                window_chunks.append(window_chunk)
                next_char = qa[i + self.window]
                next_chars.append(next_char)
                if debug:
                    print(i, 'x', window_chunk, 'y', next_char)
            if debug:
                print()

        X_shape = (len(window_chunks), self.window, len(self.chars))
        X = np.zeros(X_shape, dtype=np.bool)
        y_shape = (len(window_chunks), len(self.chars))
        y = np.zeros(y_shape, dtype=np.bool)
        for i, sentence in enumerate(window_chunks):
            # fill in reverse to accomodate shorter-than window texts
            n = -1
            for t in reversed(range(len(sentence))):
                char = sentence[t]
                X[i, n, self.char_indices[char]] = 1
                n -= 1
            # only fill out y if we have answers to train on
            if has_answer:
                y[i, self.char_indices[next_chars[i]]] = 1
        return X, y

    def split_data(self, text):
        """ Split raw data into chunks: 30% validate, 10% test, 60% train.
        """
        # remove blanks generated by split
        text_split = filter(lambda x: x.strip(), text.split(self.end_a_seq))

        # storage for splits
        train = []
        test = []
        validate = []

        # for keeping track of percentages
        total_chars = len(text)
        train_chars = 0
        test_chars = 0
        validate_chars = 0

        # here we want to split the dataset into approximate percentage
        # chunks, split on end-of-sequence markers (so the first/last
        # sentences don't get split across sample sets
        for t in text_split:
            if train_chars < total_chars * 0.6:
                train.append(t)
                train_chars += len(t)
            elif test_chars < total_chars * 0.1:
                test.append(t)
                test_chars += len(t)
            else:
                validate.append(t)
                validate_chars += len(t)

        train = (self.end_a_seq).join(train)
        test = (self.end_a_seq).join(test)
        validate = (self.end_a_seq).join(validate)

        # this will do an approximate split
        return validate, test, train

    def load_corpus(self, path):
        """ Load and vectorize our corpus, returning arrays that can
        be fed into our LSTM.
        """
        text = open(path).read().lower().replace('\n', '')

        self.chars = sorted(list(set(text)))

        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

        validate, test, train = self.split_data(text)
        val_X, val_y = self.vectorize_sentences(validate)
        ts_X, ts_y = self.vectorize_sentences(test)
        tr_X, tr_y = self.vectorize_sentences(train)
        return val_X, val_y, ts_X, ts_y, tr_X, tr_y

    def build_model(self):
        """ Build our model. This can then be used to train or load weights, etc.
        Model is built using the constants at the top of the file.
        """
        self.model = Sequential()
        # dropout breaks symmetry on zero-init weights
        self.model.add(LSTM(
            self.lstm_size,
            input_shape=(self.window, len(self.chars)),
            init='zero',
            dropout_W=self.dropout_W,
            dropout_U=self.dropout_U
        ))
        self.model.add(Dense(len(self.chars)))
        # other options include relu, tanh
        self.model.add(Activation(self.activation))
        optimizer = RMSprop(lr=self.learning_rate)
        self.model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    def train(self, X, y):
        """ Train our model for epochs, returning accuracy history
        """
        return self.model.fit(
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

    def output_from_seed(self, sentence):
        """ Take a seed sentence and generate some output from out LSTM
        """
        sentence = sentence.lower()
        # make sure our input has a end of question delim
        if sentence[-len(self.end_q_seq):] != self.end_q_seq:
            sentence += self.end_q_seq

        # this will be the entire sentence vectorized, since we're not training
        # we ignore the targets
        print("OUTPUT -------------------------------------------------")
        X, y = self.vectorize_sentences(sentence, has_answer=False)

        print('----- Generating with seed: "' + sentence + '"')

        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('-- diversity:', diversity)

            # seed the network before we attempt to collect output
            p = self.model.predict(X)
            pred_ix = self.sample(p[-1], diversity)
            pred_char = self.indices_char[pred_ix]

            generated = pred_char
            for i in range(50):
                blank = np.zeros(len(self.chars))
                last_X = np.reshape(X[-1], (1, self.window, len(self.chars)))
                trimmed_X = last_X[:, 1:]
                blank[pred_ix] = 1
                X = np.append(trimmed_X[-1], blank).reshape(1, self.window, len(self.chars))
                preds = self.model.predict(X)[0]
                pred_ix = self.sample(preds, diversity)
                pred_char = self.indices_char[pred_ix]
                generated += pred_char

            print(generated)

    def save_model(self, prefix):
        """ Write model to disk
        """
        ts = int(time.time())
        name_pfx = 'models/model_{0}_{1}'.format(self.character, prefix)
        modelname = name_pfx + '.h5'
        self.model.save(modelname)
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
                    "dropout_W": self.dropout_W,
                    "dropout_U": self.dropout_U,
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
            self.dropout_W = aux["model_params"]["dropout_W"]
            self.dropout_U = aux["model_params"]["dropout_U"]
            self.activation = aux["model_params"]["activation"]
            self.text_step = aux["model_params"]["text_step"]
            self.window = aux["model_params"]["window"]
            self.path = aux["model_params"]["path"]
            self.character = aux["model_params"]["character"]
            self.write_model = aux["model_params"]["write_model"]
        self.build_model()
        self.model.load_weights(model_h5_path)

    def test_model(self, X, y):
        """ Take a set of inputs and test our trained LSTM, returning loss
        """
        p = self.model.predict(X)
        if np.isnan(p).any():
            return 100
        score = self.model.evaluate(
            X, y,
            batch_size=self.batch_size)
        self.output_from_seed('hey jerry')
        return score

    def run(self):
        """ Generate, train and test a model, returning train, test loss
        """
        val_X, val_y, ts_X, ts_y, tr_X, tr_y, = self.load_corpus(self.path)
        self.build_model()
        history = self.train(tr_X, tr_y)
        training_loss = history.history['loss'][-1]
        test_loss = self.test_model(ts_X, ts_y)
        if self.write_model:
            self.save_model(test_loss)
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
