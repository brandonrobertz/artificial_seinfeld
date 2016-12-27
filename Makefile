corpus:
	find ./scripts/ -iname '*.shtml' -exec ./seinfeld-scripts/scrape.py  {} \; > ./seinfeld_lstm_corpus.txt

copy:
	cat seinfeld_lstm_corpus.txt | xclip -selection CLIPBOARD
