all: report slides

report:
	pdflatex report
	pdflatex report
	pdflatex report
	cp report.pdf ../d-hall_f-mlea2-tp-report.pdf

slides:
	pdflatex slides
	pdflatex slides
	pdflatex slides
	cp slides.pdf ../d-hall_f-mlea2-tp-slides.pdf

clean:
	rm -rf *.dvi *.aux *.log *.pdf *.out *.snm *.toc *.nav
