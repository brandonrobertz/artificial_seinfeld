# Artificial Seinfeld

With new episodes of Seinfeld extremely unlikely, how can we combat the prospect of watching syndicated episodes for the rest of our lives? Sit quietly, watching all 180 episodes 'til death? I think not. If Seinfeld won't make new episodes, we'll have to do it ourselves -- with the help of some AI, of course.

This repository contains all the tools you need to:
1. Scrape all Seinfeld episode transcripts
2. Pre-process the HTML files and extract all character statement/response pairs
3. Train a Long-Short Term Memory Neural Network on Seinfeld transcripts
4. Generate your own Seinfeld scripts

## How it works

The model operates on a simple principle, for each character, take their responses to
any statement/question posed. The input is the statement/question and we train on the
character's response. If we generalize this as "question/answer", we can encode each
pair like so:

    jerry i wanna tell you that meal was the worst.<q>what do you expect? it's airline food.<a>
    since when is george a writer?<q>what writer? it's a sitcom.<a>

Our model is trained by seeding the network with the first chunk of the question. The `y` target is
the next character. We continue to move this text-window forward, one character at a time, each time
supplying the next, unforseen character as the target. We do this until we get to the end-of-response
marker, `<a>`.

The full corpus is split into three chunks: 30% validation, 60% training, 10% test. During hyperparameter
optimization, we do a full model generation, training, and evaluation cycle five times, returning the
average training and test loss. The optimizer looks to minimize our test loss.

The overall theory here is that we could generate a full Seinfeld script by training a model for
each character, and then having the models generate dialogue for scenes.

## Getting Started

If you just want to train a Jerry LSTM model, you can simply use the `Makefile` to do so:

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

