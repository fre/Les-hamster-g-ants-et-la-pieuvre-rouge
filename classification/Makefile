all: classification

quiet: all

classification: donut mushroom iris optdigits tennis
	./donut.sh
	./iris.sh
	./mushroom.sh
	./optdigits.sh
	./tennis.sh

donut:
	python donut.py donut_data 2000 0.1 --noshow

iris:
	python iris.py iris_data --noshow

mushroom:
	python mushroom.py mushroom_data --noshow

optdigits:
	python optdigits.py optdigits_data_tes --noshow
	python optdigits.py optdigits_data_tra --noshow

tennis:
	python tennis.py tennis_data --noshow

clean:
	rm -rf *.bin *.eps *.svg *.pdf *.pyc
