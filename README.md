# Artificial Seinfeld

With new episodes of Seinfeld extremely unlikely, how can we combat the
prospect of watching syndicated episodes for the rest of our lives? Sit
quietly, watching all 180 episodes 'til death? I think not. If Seinfeld won't
make new episodes, we'll have to do it ourselves -- with the help of some AI,
of course.

This repository is an exploration of this idea and contains some tools to:

1.  Scrape all Seinfeld episode transcripts (based on a fork of [seinfeld-scripts](https://github.com/colinpollock/seinfeld-scripts))
2.  Extract all character statement/response pairs from the HTML transcripts
3.  Train a Long-Short Term Memory Neural Network on the Seinfeld character corpus (using [Keras](https://github.com/fchollet/keras))
4.  Generate your own Seinfeld scripts

I'm not giving up hope that NBC will pay me [$100 million to produce another
season of Seinfeld](http://www.foxnews.com/entertainment/2012/05/29/qa-former-nbc-honcho-offered-jerry-seinfeld-over-100-million-for-one-more.html)
yet. But even if they don't, this is a good way to explore character-based
language modeling and the outer fringes of Fair Use Copyright Law.

## How it works

The model operates on a simple principle: for each Seinfeld character in the
transcript corpus, take their response(s) to any statement/question posed. The
model input is the statement/question and we train on the character's response.
If we generalize this as a "question/answer" problem, we can encode each pair
like so:

    jerry i wanna tell you that meal was the worst.<q>what do you expect? it's airline food.<a>

Our model is trained by seeding the network with the first chunk of the
question. The `y` target is the next character. We continue to move this
text-window forward, one character at a time, each time supplying the next,
unforseen character as the target. We do this until we get to the
end-of-response marker, `<a>`.

To illustrate this, using the second question/answer pair, with a `WINDOW` of
10 and a batch size of 1, our inputs to our model (`X` and `y`) would look like
this:

    # first input
    X[0] = 'jerry i wa'
    y[0] = 'n'
    # second input ...
    X[1] = 'erry i wan'
    y[1] = 'n'
    # ... Nth input (end of second Q/A pair)
    X[N] = 'ne food.<a'
    y[N] = '>'

The next iteration, we'd move to the start of the next Q/A pair, fill the
window, and continue as above.  We do this until we've gone through the entire
corpus.

The full corpus is split into three chunks: 30% validation, 60% training, 10%
test. During hyperparameter optimization, we do a full model generation,
training, and evaluation cycle five times, returning the average training and
test loss. The optimizer looks to minimize our test loss.

The overall theory here is that we could generate a full Seinfeld script by
training a model for each character, then have the models feed off each other,
and generate dialogue for scenes.

## Getting Started

If you just want to train a Jerry LSTM model, you can simply use the `Makefile`
to do so:

    make

This will install dependencies (make sure you're in your virtualenv), download the
Seinology transcripts, build character corpus, train the LSTM
model, and perform a search for optimal hyperparameters.

To change the character, append the character override. This works with any of the
other make commands (below) as well:

    make CHARACTER=kramer

## Commands

The makefile contains the following commands:

    make install_deps ... install python requirements
    make scripts ........ download transcripts from seinology
    make corpus ......... extract character corpus from transcripts
                          you can change the default character, jerry,
                          by specifying another:
                              make corpus CHARACTER=kramer
    make train .......... train LSTM model using default params
    make optimize ....... perform optimal hyperparameter search
    make clean .......... delete all models, scripts, corpus, etc

# An Open Letter to Jerry Seinfeld

j/k
