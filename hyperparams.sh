#!/bin/bash
declare -a LSTM_SIZEs=(50 75 100 150 180 )
declare -a LEARNING_RATEs=("1" "0.5" "0.1" "0.03" "0.01" "0.001" "0.0001")
declare -a DROPOUTs=("0.01" "0.1" "0.2" "0.4" "0.5")
declare -a TEXT_STEPs=(1 2 5)
declare -a WINDOWs=("20" "40" "100")
declare -a ACTIVATIONs=("softmax" "relu")

export TF_CPP_MIN_LOG_LEVEL=3

echo "######################################## LEARNING_RATE ##########"
for X in "${LEARNING_RATEs[@]}"; do
    LSTM_LEARNING_RATE="$X" python lstm_text_generation.py | egrep '\*|LEARNING_RATE' \
        || ( echo "FAILURE" && exit 1 )
done

echo "######################################## LSTM_SIZE ##########"
for X in "${LSTM_SIZEs[@]}"; do
    LSTM_SIZE="$X" python lstm_text_generation.py | egrep "\*|LSTM_SIZE" \
        || ( echo "FAILURE" && exit 1 )
done

echo "######################################## DROPOUT ##########"
for X in "${DROPOUTs[@]}"; do
    LSTM_DROPOUT="$X" python lstm_text_generation.py | egrep '\*|DROPOUT' \
        || ( echo "FAILURE" && exit 1 )
done

echo "######################################## TEXT_STEP ##########"
for X in "${TEXT_STEPs[@]}"; do
    LSTM_TEXT_STEP="$X" python lstm_text_generation.py | egrep '\*|TEXT_STEP' \
        || ( echo "FAILURE" && exit 1 )
done

echo "######################################## WINDOW ##########"
for X in "${WINDOWs[@]}"; do
    LSTM_WINDOW="$X" python lstm_text_generation.py | egrep '\*|WINDOW' \
        || ( echo "FAILURE" && exit 1 )
done

echo "######################################## ACTIVATION ##########"
for X in "${ACTIVATIONs[@]}"; do
    LSTM_ACTIVATION="$X" python lstm_text_generation.py | egrep '\*|ACTIVATION' \
        || ( echo "FAILURE" && exit 1 )
done
