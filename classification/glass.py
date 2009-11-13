# glass.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import pickle
import sys
import shuffle
from matplotlib import pylab

filename = "glass_data"

noshow = 0  # Do not show the pylab window

if sys.argv[len(sys.argv) - 1] == '--noshow':
    noshow = 1

if len(sys.argv) >= 2:
    filename = sys.argv[1]

def tr(val):
    names = {'1': "building_windows_float_processed",
             '2': "building_windows_non_float_processed",
             '3': "vehicle_windows_float_processed",
             '4': "vehicle_windows_non_float_processed",
             '5': "containers",
             '6': "tableware",
             '7': "headlamps"}
    return names[val]

def glass(filename):
    fh = open(filename, "r")

    data = []
    labels = []

    fl = fh.readlines()
    for line in fl:
        vals = line.split('\n')[0].rsplit(',')
        if len(vals) != 11:
            break
        data.append([float(vals[i + 1]) for i in range(9)])
        labels.append(tr(vals[10]))

    shuffle.shuffle(data, labels, shuffle.generate_seed(len(data)))

    return numpy.array(data), labels

def __work():
    print "Reading", filename

    o = glass(filename)

    pickle.dump(o, open(filename + ".bin", "w"))

    print "Written", filename + ".bin"

# Comment to do nothing
__work()
