# Artificial Jerry Seinfeld

With new episodes of Seinfeld extremely unlikely, how can we combat the prospect of watching syndicated episodes for the rest of our lives? Sit quietly, watching all 180 episodes 'til death. I think not. If Seinfeld won't make new episodes, we'll have to do it ourselves -- with AI, of course.

This repository contains all the tools you need to:
1. Scrape all Seinfeld episode transcripts
2. Pre-process the HTML files and extract all character statement/response pairs
3. Train a Long-Short Term Memory Neural Network on Seinfeld transcripts
4. Generate your own Seinfeld scripts

## Getting Started

If you just want to train a Jerry LSTM model, you can simply use the `Makefile` to do so:

    make

This will install dependencies (`pip install -r requirements.txt`, make sure you're in your
virtualenv, etc), download the Seinology transcripts, build character corpus, train the LSTM
model, and perform a search for optimal hyperparameters.

## Makefile

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

