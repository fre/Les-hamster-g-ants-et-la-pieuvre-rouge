all: classification

quiet: all

images:
	mkdir images

datasets: images donut mushroom iris optdigits tennis krkopt glass donut_simple linear linear_simple xor xor_simple

classification: datasets
	./donut.sh
	./linear.sh
	./iris.sh
	./mushroom.sh
	./optdigits.sh
	./tennis.sh
	./krkopt.sh
	./glass.sh

donut:
	python donut.py donut_data 2000 0.1 --noshow

donut_simple:
	python donut.py donut_simple_data 1000 0.0 --noshow

xor:
	python xor.py xor_data 2000 0.1 --noshow

xor_simple:
	python xor.py xor_simple_data 1000 0.0 --noshow

linear:
	python linear.py linear_data 2000 0.1 --noshow

linear_simple:
	python linear.py linear_simple_data 1000 0.0 --noshow

iris:
	python iris.py iris_data --noshow

krkopt:
	python krkopt.py krkopt_data --noshow

glass:
	python glass.py glass_data --noshow

mushroom:
	python mushroom.py mushroom_data --noshow

optdigits:
	python optdigits.py optdigits_data_tes --noshow
	python optdigits.py optdigits_data_tra --noshow

tennis:
	python tennis.py tennis_data --noshow

clean:
	rm -rf *.bin *.eps *.svg *.pdf *.pyc images *.dot
