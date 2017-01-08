CHARACTER=jerry
MONGOHOST=
MONGODB=

default: install_deps scripts corpus optimize

install_deps:
	pip install -r requirements.txt

scripts:
	mkdir scripts 2> /dev/null
	./download.py ./scripts/

corpus:
	find ./scripts/ -iname '*.shtml' -exec ./scrape.py --character ${CHARACTER} {} \; \
			> ./seinfeld_lstm_corpus.${CHARACTER}.txt

train:
	python seinfeld_lstm.py

optimize:
	python optimize.py ${CHARACTER} ${MONGOHOST} ${MONGODB}

clean:
	find ./ -iname "*.pyc" -delete
	rm -rf scripts
	rm seinfeld_lstm_corpus.txt
	rm *.p
	rm *.h5

mongo:
	./mongodb-linux-x86_64-3.4.1/bin/mongod --dbpath ./db/ --port 1234
