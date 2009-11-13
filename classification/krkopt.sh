#!/bin/sh

for size in 100 200 500 1000 2000
do
time python classification.py krkopt_data 'e' $size --noshow
done
