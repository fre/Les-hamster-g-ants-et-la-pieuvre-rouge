#!/bin/sh

for size in 100 200 500 1000 2000 0
do
time python classification.py mushroom_data 'e' $size --noshow
done
