#!/usr/bin/env python2.7
from __future__ import print_function
from seinfeld_lstm import SeinfeldAI
import sys
import settings
import argparse


def parse_args():
    desc = 'Read input (a statement or question) from stdin and generate ' \
        'outputs using a given model and (optional) character name.'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('model', type=str,
                        help='Path to saved HDF5 (.h5) model.')
    parser.add_argument('--character',
                        help='An identifier for the output. This is an ' \
                        'optional parameter that controls whether, and what ' \
                        'if selected, to prefix the output with. E.g., ' \
                        'choosing --character Jerry, will make the model ' \
                        'prefix outputs with Jerry: [model output.')
    parser.add_argument('--temp', type=float, default=settings.TEMPERATURE,
                        help='Softmax activation "temperature". 0.0 through ' \
                        '1.0. Higher values will allow the model to make ' \
                        'more diverse, riskier predictions.')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    model = SeinfeldAI(debug=False)
    model.load_model(args.model)

    delim = settings.END_Q_SEQ
    if delim == '|':
        delim = '\\' + settings.END_Q_SEQ

    output_until = delim + '|' + '\\.'

    for line in sys.stdin:
        if settings.END_A_SEQ not in line:
            line += settings.END_A_SEQ
        if args.character:
            print(args.character + ': ', end='')
        model.output_from_seed(
            line, temperature=args.temp, max_chars=0,
            output_until=output_until
        )
