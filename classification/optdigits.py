# optdigits.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import pickle
import sys
import shuffle
from matplotlib import pylab

filename = "optdigits_data_tra"

noshow = 0  # Do not show the pylab window

if sys.argv[len(sys.argv) - 1] == '--noshow':
    noshow = 1

if len(sys.argv) >= 2:
    filename = sys.argv[1]

def optdigits(filename):
    fh = open(filename, "r")

    data = []
    labels = []

    fl = fh.readlines()
    for line in fl:
        vals = line.split('\n')[0].rsplit(',')
        if len(vals) != 65:
            break
        data.append([int(vals[i]) for i in range(64)])
        labels.append(vals[64])

    shuffle.shuffle(data, labels, shuffle.generate_seed(len(data)))

    return numpy.array(data), labels

def __work():
    print "Reading", filename

    o = optdigits(filename)

    pickle.dump(o, open(filename + ".bin", "w"))

    print "Written", filename + ".bin"

# Comment to do nothing
__work()
