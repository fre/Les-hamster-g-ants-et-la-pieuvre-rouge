all:
	make -C decision

quiet:
	make -C decision quiet

clean:
	make -C decision clean
	make -C classification clean
	make -C clustering clean
	make -C doc clean

dist:
	git archive --format=tar --prefix=d-hall_f-mlea2-tp1/ HEAD | gzip > d-hall_f-mlea2-tp1.tar.gz

report:
	make -C doc

quickreport:
	make -C doc report
