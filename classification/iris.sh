#!/bin/sh

for size in 0
do
time python classification.py iris_data '' $size --noshow
done
