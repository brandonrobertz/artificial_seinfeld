CHARACTER=jerry

default: install_deps scripts corpus train optimize

install_deps:
	pip install -r requirements.txt

scripts:
	mkdir scripts 2> /dev/null
	./download.py ./scripts/

corpus:
	find ./scripts/ -iname '*.shtml' -exec ./scrape.py  --character ${CHARACTER} {} \; \
			> ./seinfeld_lstm_corpus.${CHARACTER}.txt

train:
	python lstm_text_generation.py

optimize:
	python optimize.py

clean:
	find ./ -iname "*.pyc" -delete
	rm -rf scripts
	rm seinfeld_lstm_corpus.txt
	rm *.p
	rm *.h5
