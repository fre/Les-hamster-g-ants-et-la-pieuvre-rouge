all: figures report slides

figures:
	cd ../decision && make quiet
	cd ../classification && make quiet
	cd ../clustering && make quiet
	cp "../clustering/teddy_toy_data_0_clustering_K-MEANSPP(16) (max distance)_Start.eps" "../clustering/teddy_toy_data_0_clustering_K-MEANSPP(16)_(max_distance)_Start.eps"
	cp "../clustering/teddy_toy_data_0_clustering_K-MEANSPP(16) (max distance)_Start.pdf" "../clustering/teddy_toy_data_0_clustering_K-MEANSPP(16)_(max_distance)_Start.pdf"
	cp "../clustering/teddy_toy_data_0_clustering_K-MEANSPP(16) (max distance)_Start.svg" "../clustering/teddy_toy_data_0_clustering_K-MEANSPP(16)_(max_distance)_Start.svg"

report:
	pdflatex report
	pdflatex report
	pdflatex report
	cp report.pdf ../d-hall_f-mlea2-tp1-report.pdf

slides:
	pdflatex slides
	pdflatex slides
	pdflatex slides
	cp slides.pdf ../d-hall_f-mlea2-tp1-slides.pdf

clean:
	rm -rf *.dvi *.aux *.log *.pdf *.out *.snm *.toc *.nav
