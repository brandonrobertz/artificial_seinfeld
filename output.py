#!/usr/bin/env python2.7
from seinfeld_lstm import SeinfeldAI
import sys

s = SeinfeldAI(sys.argv[1])
model = s.load_model()
s.output_from_seed(model, 'hello jerry<q>')
s.output_from_seed(model, 'george, you\'re a dick<q>')
