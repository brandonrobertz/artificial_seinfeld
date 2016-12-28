corpus:
	find ./scripts/ -iname '*.shtml' -exec ./scrape.py  {} \; > ./seinfeld_lstm_corpus.txt

download:
	./download.py ./scripts/

copy:
	cat seinfeld_lstm_corpus.txt | xclip -selection CLIPBOARD
