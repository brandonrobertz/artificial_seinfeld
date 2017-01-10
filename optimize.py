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

import settings

# this needs to go before keras loads to disable annoying verbose logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa

from seinfeld_lstm import SeinfeldAI


space = {
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(0.5)),
    'lstm_size': hp.quniform('lstm_size', 600, 2000, 1),
    'dropout_W': hp.uniform('dropout_W', 0.0, 0.9),
    'dropout_U': hp.uniform('dropout_U', 0.0, 0.9),
    'window': hp.quniform('window', 10, 80, 1),
    'epochs':  hp.quniform('epochs', 10, 100, 1)
    # 'activation': hp.choice('activation', ['softmax', 'relu', 'tanh'])
}

# trials = None
trials = None


def save_trials(character):
    paramsfile = "lstm_hyperopt.{0}.p".format(character)
    pickle.dump(trials, open(paramsfile, "wb"))


def main(character):
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
        # args['path'] = 'seinfeld_lstm_corpus.{0}.txt'.format(character)
        args['batch_size'] = 25
        try:
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

    for i in range(100):
        best = fmin(
            objective,
            space=space,
            algo=tpe.suggest,
            max_evals=(i+1),
            trials=trials)
        print()
        print()
        print(i, "Final trials is:",
              np.sort(np.array([
                  x for x in trials.losses() if x is not None])))
        print()
        print(i, "Best result was: ", best)
        save_trials(character)


def signal_handler(signal, frame):
    print()
    print()
    print("Final trials is:", np.sort(-1*np.array(trials.losses())))
    save_trials()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    argc = len(sys.argv)
    if argc > 4 or ((argc == 2) and sys.argv[1] == "-h"):
        print("USAGE ./optimize.py CHARACTER [mongo_host:port] [mogo_db]")
        print("    CHARACTER - must be the name of a character with spoken")
        print("      line in the source transcripts")
        print("    mongo_host:port - host:port of mongo instance for parallel")
        print("      searches. if omitted, optimize will run in standalone")
        print("      search mode")
        print("    mongo_db will default to seinfeld_db if blank")
        sys.exit(1)

    character = sys.argv[1]

    if argc >= 3:
        mongo_host = sys.argv[2]
        mongo_db = sys.argv[3] if len(sys.argv) == 4 else 'seinfeld_db'
        mongo_connect = 'mongo://{0}/{1}/jobs'.format(mongo_host, mongo_db)
        print("Setting up distributed search on mongo instance {0}/{1}".format(
            mongo_host, mongo_db))
        trials = MongoTrials(mongo_connect)
    else:
        print("Using standalone (single-machine) search mode")

    main(character)
