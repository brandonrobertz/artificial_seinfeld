corpus:
	find ./scripts/ -iname '*.shtml' -exec ./scrape.py  {} \; > ./seinfeld_lstm_corpus.txt

download:
	./download.py ./scripts/

copy:
	cat seinfeld_lstm_corpus.txt | xclip -selection CLIPBOARD

clean_transform:
	cat seinfeld_full_corpus.sentences.txt \
	| sed 's/\?/\?\n/g' \
	| sed 's/\!/\!\n/g' \
	| tr -d \.\?\! \
	| tr \\n \  \
	| sed 's/\s\{2,\}/ /g' \
	| fold -s \
	> seinfeld_full_corpus.sentences.cleaned.txt

clean:
	cat seinfeld_lstm_corpus.txt \
		| tr -c '[A-Za-z0-9\n ]' ' '

train:
	./stateful_lstm_text_generation.py seinfeld_lstm_corpus.txt
