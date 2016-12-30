#!/usr/bin/env python2.7
'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import json

LSTM_SIZE = 128 * 2
EPOCHS = 10
BATCH_SIZE = 128
TEXT_STEP = 3

if len(sys.argv) < 2:
    print( "USAGE: {0} path_to_corpus".format(sys.argv[0]))
    sys.exit(1)

# path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
text = open(sys.argv[1]).read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# maxlen = 40
maxlen = max(map(lambda x: len(x), text.split('\n')))
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, TEXT_STEP):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('LSTM_SIZE',LSTM_SIZE)
print('EPOCHS',EPOCHS)
print('BATCH_SIZE',BATCH_SIZE)
print('TEXT_STEP',TEXT_STEP)

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: a single LSTM
print('Build model...')
model = Sequential()
model.add(LSTM(LSTM_SIZE, input_shape=(maxlen, len(chars))))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.02)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

# train the model, output generated text after each iteration
# for iteration in range(1, 10):
#     print()
#     print('-' * 50)
#     print('Iteration', iteration)
model.fit(X, y, batch_size=BATCH_SIZE, nb_epoch=EPOCHS)
start_index = random.randint(0, len(text) - maxlen - 1)

# for diversity in [0.2, 0.5]:  # , 1.0, 1.2]:
#     print()
#     print('----- diversity:', diversity)

# generated = ''
# sentence = text[start_index: start_index + maxlen]
# generated += sentence
# print('----- Generating with seed: "' + sentence + '"')
# sys.stdout.write(generated)

generated = 'what\'s going on?<q>'
sentence = generated
diversity = 0.8
for i in range(140):
    x = np.zeros((1, maxlen, len(chars)))
    for t, char in enumerate(sentence):
        x[0, t, char_indices[char]] = 1.

    preds = model.predict(x, verbose=0)[0]
    next_index = sample(preds, diversity)
    next_char = indices_char[next_index]

    generated += next_char
    sentence = sentence[1:] + next_char

    sys.stdout.write(next_char)
    sys.stdout.flush()
print('OUTPUT:',generated)
print()

model.save('lstm_window_model.h5')
with open('lstm_window_char_indices.json','w') as f:
    f.write(json.dumps(char_indices))
