from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
# import walker  # walker.py in the same folder
import numpy as np
import signal
import sys
import pickle

from lstm_text_generation import ArtificialJerry

space = {
    # Learning rate should be between 0.00001 and 1
    'learning_rate': hp.loguniform('learning_rate', np.log(1e-5), np.log(1)),
    'lstm_size': hp.quniform('lstm_size', 20, 300, 1),
    'dropout': hp.quniform('dropout', 0.0, 1, 1),
    'activation': hp.choice('activation', ['softmax', 'relu']),
    'text_step': hp.quniform('text_step', 1, 5, 1),
    'window': hp.quniform('window', 5, 140, 1)
    # 'layer1_size': hp.quniform('layer1_size', 10, 100, 1),
    # 'layer2_size': hp.quniform('layer2_size', 10, 100, 1),
    # 'layer3_size': hp.quniform('layer3_size', 10, 100, 1),
    # 'future_discount_max': hp.uniform('future_discount_max', 0.5, 0.99),
    # 'future_discount_increment': hp.loguniform(
    #     'future_discount_increment',
    #     np.log(0.001),
    #     np.log(0.1)),
    # 'recall_memory_size': hp.quniform('recall_memory_size', 1, 100, 1),
    # 'recall_memory_num_experiences_per_recall':  hp.quniform(
    #     'recall_memory_num_experiences_per_recall',
    #     10,
    #     2000,
    #     1),
    # 'num_epochs':  hp.quniform('num_epochs', 1, 10, 1),
}

trials = None


def save_trials():
    pickle.dump(trials, open("lstm_hyperopt.p", "wb"))


def main():
    epochs = 3
    global trials
    try:
        trials = pickle.load(open("lstm_hyperopt.p", "rb"))
    except:
        print("Starting new trials file")
        trials = Trials()

    def objective(args):
        print("args:", args)
        jerry = ArtificialJerry(**args)
        tr_err = np.zeros(epochs)
        ts_err = np.zeros(epochs)
        for i in range(epochs):
            training_loss, test_loss = jerry.epoch()
            tr_err[i] = training_loss
            ts_err[i] = test_loss
        print("Train:", tr_err.mean(), "Test:", ts_err.mean(), "for", args)
        return {'loss': -ts_err.mean(), 'status': STATUS_OK}

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
              np.sort(-1*np.array([
                  x for x in trials.losses() if x is not None])))
        print()
        print(i, "Best result was: ", best)
        save_trials()


def signal_handler(signal, frame):
    print()
    print()
    print("Final trials is:", np.sort(-1*np.array(trials.losses())))
    save_trials()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    main()
