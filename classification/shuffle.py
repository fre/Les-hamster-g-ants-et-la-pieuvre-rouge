# shuffly.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math

def generate_seed(max):
    return numpy.random.random_integers(0, max - 1, max)

def shuffle(data, labels, sw):

    max = len(data)

    i = 0
    # swap
    while i < max:
        j = sw[i]
        tmp = data[i]
        data[i] = data[j]
        data[j] = tmp
        tmp = labels[i]
        labels[i] = labels[j]
        labels[j] = tmp
        i += 1

    return data, labels
