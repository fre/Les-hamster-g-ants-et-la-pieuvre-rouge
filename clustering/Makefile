all: clustering

quiet: all

tiny_toy:
	python tiny_toy.py tiny_toy_data --noshow

teddy_toy:
	python teddy_toy.py teddy_toy_data --noshow
	python teddy_toy.py teddy_toy_small_data 2 2 20 200 5 --noshow

clustering: tiny_toy teddy_toy
	./tiny_toy.sh
	./teddy_toy.sh

clean:
	rm -rf *.bin *.eps *.svg *.pdf *.pyc
