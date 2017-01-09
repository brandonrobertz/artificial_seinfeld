#!/usr/bin/env python2.7
from seinfeld_lstm import SeinfeldAI
import sys
import settings


def usage():
    print('USAGE: output.py path/to/model.h5')
    print('  this reads input from stdin by default')
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        usage()

    h5_path = sys.argv[1]

    model = SeinfeldAI()
    model.load_model(h5_path)

    for line in sys.stdin:
        if settings.END_A_SEQ not in line:
            line += settings.END_A_SEQ
        model.output_from_seed(line)
