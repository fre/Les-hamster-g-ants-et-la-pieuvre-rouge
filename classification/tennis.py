# tennis.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import pickle
import sys
import shuffle
from matplotlib import pylab

filename = "tennis_data"

noshow = 0  # Do not show the pylab window

if sys.argv[len(sys.argv) - 1] == '--noshow':
    noshow = 1

if len(sys.argv) >= 2:
    filename = sys.argv[1]

def tennis_dicts(filename):
    fh = open(filename, "r")

    count = 0
    dicts = []
    for i in range(5):
	dicts.append({})

    fl = fh.readlines()
    for line in fl:
        vals = line.split('\n')[0].rsplit(',')
        if len(vals) != 5:
            break
        for i in range(5):
            if vals[i] in dicts[i]:
                dicts[i][vals[i]] += 1
            else:
                dicts[i][vals[i]] = 1
        count += 1

    return count, dicts

def tennis(filename, count, dicts, seed):
    fh = open(filename, "r")

    data = []
    labels = []

    fl = fh.readlines()
    for line in fl:
        vals = line.split('\n')[0].rsplit(',')
        if len(vals) != 5:
            break
        element = []
        for i in range(4):
            v = 0
            keys = dicts[i].keys()
            keys.sort()
            for k in keys:
                if vals[i] == k:
                    element.append(v)
                v += 1
        data.append(element)
        labels.append(vals[4])

    shuffle.shuffle(data, labels, seed)

    return numpy.array(data, numpy.int16), labels

def __work():
    print "Reading", filename

    count, dicts = tennis_dicts(filename)
    seed = shuffle.generate_seed(count)

    o = tennis(filename, count, dicts, seed)
    pickle.dump(o, open(filename + ".bin", "w"))
    print "Written", filename + ".bin"

# Comment to do nothing
__work()
