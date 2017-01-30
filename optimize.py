#!/usr/bin/env python2.7
from __future__ import print_function
from hyperopt import fmin, tpe, hp, STATUS_OK, STATUS_FAIL, Trials
from hyperopt.mongoexp import MongoTrials
# import walker  # walker.py in the same folder
import numpy as np
import signal
import sys
import pickle
import os
import argparse

import settings

# this needs to go before keras loads to disable annoying verbose logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa

from seinfeld_lstm import SeinfeldAI


space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(0.5)),
    'lstm_size': hp.quniform('lstm_size', 600, 2000, 1),
    'dropout_W': hp.uniform('dropout_W', 0.0, 0.5),
    'dropout_U': hp.uniform('dropout_U', 0.0, 0.5),
    'window': hp.quniform('window', 10, 80, 1),
    'epochs':  hp.quniform('epochs', 10, 20, 1)
    # 'activation': hp.choice('activation', ['softmax', 'relu', 'tanh'])
}

# trials = None
trials = None


def save_trials(character):
    paramsfile = "lstm_hyperopt.{0}.p".format(character)
    pickle.dump(trials, open(paramsfile, "wb"))


def main(character, corpus):
    epochs = 1
    global trials
    paramsfile = "lstm_hyperopt.{0}.p".format(character)
    try:
        trials = pickle.load(open(paramsfile, "rb"))
    except:
        print("Starting new trials file")
        trials = Trials()

    def objective(args):
        args['character'] = character
        args['path'] = corpus #'seinfeld_lstm_corpus.{0}.txt'.format(character)
        args['batch_size'] = 300
        try:
            print('Running with args', args)
            model = SeinfeldAI(**args)
            tr_err = np.zeros(epochs)
            ts_err = np.zeros(epochs)
            for i in range(epochs):
                training_loss, test_loss = model.run()
                tr_err[i] = training_loss
                ts_err[i] = test_loss
            print("Train:", tr_err.mean(), "Test:", ts_err.mean(), "for", args)
        except Exception, e:
            print("Error running model", e)
            return {'status': STATUS_FAIL}
        return {'loss': ts_err.mean(), 'status': STATUS_OK}

    for i in range(300):
        best = fmin(
            objective,
            space=space,
            algo=tpe.suggest,
            max_evals=(i+1),
            trials=trials
        )
        print()
        print()
        print(i, "Final trials is:",
              np.sort(np.array([
                  x for x in trials.losses() if x is not None])))
        print()
        print(i, "Best result was: ", best)
        save_trials(character)


def parse_args():
    desc = 'Search for optimal hyperparams for a given corpus, saving ' \
        'models as we go.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('character', type=str,
                        help='An identifier for the model.')
    parser.add_argument('corpus', type=str,
                        help='The path to the corpus to train on.')
    parser.add_argument('--mongohost', type=str,
                        help='Mongo DB hostname:port, for distributed search')
    parser.add_argument('--mongodb', type=str,
                        help='Mongo DB db name, for distributed search')
    args = parser.parse_args()
    return args


def signal_handler(signal, frame):
    print()
    print()
    print("Final trials is:", np.sort(-1*np.array(trials.losses())))
    save_trials()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    args = parse_args()

    if args.mongohost and args.mongodb:
        mongo_connect = 'mongo://{0}/{1}/jobs'.format(
            args.mongohost, args.mongodb)
        print("Using distributed search on mongo instance {0}".format(
            mongo_connect))
        trials = MongoTrials(mongo_connect)
    else:
        print("Using standalone (single-machine) search mode")

    main(args.character, args.corpus)
