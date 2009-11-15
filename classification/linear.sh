#!/bin/sh

for size in 100 500 1000 0
do
time python classification.py linear_data 'inside' $size --noshow
time python classification.py linear_simple_data 'inside' $size --noshow
done
