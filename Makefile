all:
	make -C classification

quiet:
	make -C classification quiet

clean:
	make -C classification clean
	make -C clustering clean
	make -C doc clean

dist:
	git archive --format=tar --prefix=d-hall_f-mlea2-tp1/ HEAD | gzip > d-hall_f-mlea2-tp.tar.gz

report:
	make -C doc

quickreport:
	make -C doc report
