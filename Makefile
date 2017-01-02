default: requirements download corpus train optimize

requirements:
	pip install -r requirements.txt

download:
	./download.py ./scripts/

corpus:
	find ./scripts/ -iname '*.shtml' -exec ./scrape.py  {} \; > ./seinfeld_lstm_corpus.txt

train:
	lstm_text_generation.py

optimize:
	python optimize.py
