#!/bin/sh

for size in 100 200 500 1000 2000 0
do
time python classification.py optdigits_data_tes '' $size --noshow
time python classification.py optdigits_data_tra '' $size --noshow
done
