download:
	./download.py ./scripts/

corpus:
	find ./scripts/ -iname '*.shtml' -exec ./scrape.py  {} \; > ./seinfeld_lstm_corpus.txt

copy:
	cat seinfeld_lstm_corpus.txt | xclip -selection CLIPBOARD
