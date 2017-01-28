CHARACTER=jerry
MONGOHOST=
MONGODB=
CORPUS=

default: install_deps scripts summaries corpus optimize

install_deps:
	pip install -r requirements.txt

download_scripts:
	mkdir -p scripts 2> /dev/null
	./download.py ./scripts/

download_summaries:
	mkdir -p summaries 2> /dev/null
	./download.py --summary ./summaries/

corpus:
	find ./scripts/ -iname '*.shtml' -exec ./scrape.py --character ${CHARACTER} {} \; \
			> ./seinfeld_lstm_corpus.${CHARACTER}.txt

summary_corpus:
	find ./summaries/ -iname '*.shtml' -exec ./scrape.py --mode synopsis {} \; \
			> ./synopsis_corpus.txt

summaries: download_summaries summary_corpus

train:
	python seinfeld_lstm.py

optimize:
	python optimize.py ${CHARACTER} ${CORPUS} # ${MONGOHOST} ${MONGODB}

clean:
	find ./ -iname "*.pyc" -delete
	rm -rf scripts
	rm -rf summaries
	rm seinfeld_lstm_corpus.txt
	rm *.p
	rm *.h5

mongo:
	./mongodb-linux-x86_64-3.4.1/bin/mongod --dbpath ./db/ --port 1234
