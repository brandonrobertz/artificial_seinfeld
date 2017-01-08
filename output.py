#!/usr/bin/env python2.7
from seinfeld_lstm import SeinfeldAI
import sys
import settings


def usage():
    print('USAGE: output.py path/to/model.h5')
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()

    jerry = SeinfeldAI(character="jerry")
    jerry.load_model(sys.argv[1])
    jerry.output_from_seed('hello jerry' + settings.END_Q_SEQ)
    jerry.output_from_seed('george you\'re a dick'  + settings.END_Q_SEQ)
