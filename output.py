#!/usr/bin/env python2.7
from __future__ import print_function
from seinfeld_lstm import SeinfeldAI
import sys
import settings


def usage():
    print('USAGE: output.py path/to/model.h5 [temperature]')
    print('  this reads input from stdin line-by-line')
    print('  temperature - float, part of softmax activation (0.3 default)')
    sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        usage()

    h5_path = sys.argv[1]

    temperature = 0.3
    if len(sys.argv) == 3:
        temperature = float(sys.argv[2])

    model = SeinfeldAI()
    model.load_model(h5_path)

    for line in sys.stdin:
        if settings.END_A_SEQ not in line:
            line += settings.END_A_SEQ
        model.output_from_seed(line, temperature=temperature)
