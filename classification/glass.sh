#!/bin/sh

for size in 0
do
time python classification.py glass_data '' $size --noshow
done
