#!/bin/sh

for size in 0
do
time python classification.py tennis_data 'Yes' $size --noshow
done
