#!/usr/bin/env python2.7
from seinfeld_lstm import SeinfeldAI
import sys


def usage():
    print('USAGE: output.py path/to/model.h5')
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()

    s = SeinfeldAI()
    model = s.load_model(sys.argv[1])
    s.output_from_seed(model, 'hello jerry<q>')
    s.output_from_seed(model, 'george you\'re a dick<q>')
