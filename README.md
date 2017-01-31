# Artificial Seinfeld

With new episodes of Seinfeld extremely unlikely, how can we combat the
prospect of watching syndicated episodes for the rest of our lives? Sit
quietly, watching all 180 episodes 'til death? I think not. If Seinfeld won't
make new episodes, we'll have to do it ourselves -- but we'll definitely need
some AI to do the actual writing. Nobody wants to do that much work.

This repository is an exploration of this idea and contains some tools to:

1.  Scrape all Seinfeld episode transcripts (based on a fork of [seinfeld-scripts](https://github.com/colinpollock/seinfeld-scripts))
2.  Extract all character statement/response pairs from the HTML transcripts
3.  Train a Long-Short Term Memory Neural Network on the Seinfeld character corpus (using [Keras](https://github.com/fchollet/keras))
4.  Generate your own Seinfeld scripts

I'm not giving up hope that NBC will pay me [$100 million to produce another
season of Seinfeld](http://www.foxnews.com/entertainment/2012/05/29/qa-former-nbc-honcho-offered-jerry-seinfeld-over-100-million-for-one-more.html). Or maybe they'll go the other
way, fail to see humor in this very complex joke. Either way,
this is a good way to explore character-based language modeling and the outer
fringes of Fair Use Copyright Law.

## How it works

The model operates on a simple principle: for each Seinfeld character in the
transcript corpus, take their response(s) to any statement/question posed. Then
we can orchestrate models from different characters, creating new scenes and episodes.

The model input is a statement/comment/question and we train on the character's response.
If we generalize this as a "question/answer" problem, we can encode each pair
like so:

    jerry i wanna tell you that meal was the worst.<q>what do you expect? it's airline food.<a>

Our model is trained by seeding the network with the first chunk of the
question. The `y` target is the next character. We continue to move this
text-window forward, one character at a time, each time supplying the next,
unforseen character as the target. We do this until we get to the
end-of-response marker, `<a>`. (Note that `<a>` is an example and the actual
implementation uses a single-character marker, which is removed from the corpus input
in preprocessing.)

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
test loss. The optimizer looks to minimize our test, not train, loss to
avoid overfitting. We also use dropout to break symmetry and improve model
generalization.

The overall theory here is that we could generate a full Seinfeld script by
training a model for each character, training a model to develop a synopsis and
outline, and then gather the outputs into a series of scenes.


## Getting Started - Character Models

If you just want to train a Jerry LSTM model, you can simply use the `Makefile`
to do so:

    make

This will install dependencies (make sure you're in your virtualenv), download the
Seinology transcripts, build character corpus, train the LSTM
model, and perform a search for optimal hyperparameters.

To change the character, append the character override. This works with any of the
other make commands (below) as well:

    make CHARACTER=kramer


### Synopsis Model

Just having a question-answer for each character isn't enough. We need some form of
structure and plotline. Luckily for us, Seinfeld episodes are typically named after
an object, place, or short action that appears in the episode. We exploit this to
build a synopsis-generation model that takes a short input and outputs a
synopsis. To get started, use the following command to build the synposis corpus:

    make summaries

This will download all the "episode guides" from Seinology and will build a
title/synopsis corpus in the same format as statement/response for character
models.

Once we have this built, we can train a synopsis model on it:

    make optimize CHARACTER=synopsis CORPUS=./synopsis_corpus.txt


## Settings

While the python scripts take command line arguments, it gets annoying keeping
track of all the default params and specifying them everywhere. To make this
easier, the default params, as well as some non-specified params, such as
start/end sequences, are given in settings.py.


## Output

Once we have a trained model, in this example let's use our trained synopsis
model, we can use the output command to generate outputs like so:

    echo "the bottom feeder#" \
    | ./output.py --character Synopsis --temp 0.7 ./models/model_synopsis_1.48.h5

        Synopsis: george discovers that he hears to broke at the pired to him 
        man be nemeves the break up with the one of cake he makes a job ex a 
        steel to to frank the reest uncle from the sane takes to mivi seod of 
        george meets discovers the real pronices.

## Examples


Total 4 (delta 3), reused 0 (delta 0)                                                                                                (seinfeld)brando@OSX:~/seinfeld$ echo hey jerry \
              | ./output.py models/model_1483862114.h5 jerry 0.3 \
              2> /dev/null
          yeah i was something. i don't know he look like the car with the time.

## Commands

The makefile contains the following commands:

    make install_deps ....... install python requirements
    make download_scripts ... download transcripts from seinology
    make character_corpus ... extract character corpus from transcripts
        you can change the default character, jerry, by specifying another:
        make corpus CHARACTER=kramer
        character options: elaine, george, kramer, newman, jerry
    make train .............. train LSTM model using default params
    make optimize ........... perform optimal hyperparameter search
        this accepts MONGOHOST=localhost:1234 and MONGODB=seinfeld_job after make
        optimize to enable distributed searches. use CHARACTER and CORPUS
        options to change character label and corpus location (optional)
    make clean .............. delete all models, scripts, corpus, etc
    make download_summaries .. download the episode summaries
    make summary_corpus ...... compile summaries into a title/synopsis corpus for training
    make summaries ........... wrapper for download + compile synopsis corpus

By default these commands will use the Jerry corpus and will label the model 'jerry'.
You can override corpus location by appending `CORPUS=/corpus/location.txt` to your
`make` commands.

# An Open Letter to Jerry Seinfeld

j/k
