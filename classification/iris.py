# iris.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import pickle
import sys
import shuffle
from matplotlib import pylab

filename = "iris_data"

noshow = 0  # Do not show the pylab window

if sys.argv[len(sys.argv) - 1] == '--noshow':
    noshow = 1

if len(sys.argv) >= 2:
    filename = sys.argv[1]

def iris(filename):
    fh = open(filename, "r")

    data = []
    labels = []

    fl = fh.readlines()
    for line in fl:
        vals = line.split('\n')[0].rsplit(',')
        if len(vals) != 5:
            break
        data.append([float(vals[i]) for i in range(4)])
        labels.append(vals[4])

    shuffle.shuffle(data, labels, shuffle.generate_seed(len(data)))

    return numpy.array(data), labels

def display(data, labels):
    i = 0

    ls = {}
    for l in labels:
        ls[l] = [[] for i in range(len(data[0]))]
    keys = sorted(ls.keys())

    for e in range(len(data)):
        for k in keys:
            if labels[e] == k:
                ar = ls[k]
                for i in range(len(ar)):
                    ar[i].append(data[e][i])

    ind = range(len(data[0]))
    width = 0.8 / len(keys)
    fig = pylab.figure()
    ax = fig.add_subplot(111)

    colors = [['r', 'orange'], ['g', 'magenta'], ['b', 'cyan']]

    for i in range(len(keys)):
        avg = [numpy.mean(ls[keys[i]][k]) for k in ind]
        dev = [numpy.std(ls[keys[i]][k]) for k in ind]
        indx = [k + width * i for k in ind]
        ax.bar(indx, avg, width, yerr=dev, label=keys[i],
               color=colors[i][0], ecolor=colors[i][1])

    ax.set_ylabel('value')
    ax.set_xlabel('component index')
    ax.set_title('Average component values for Iris types')
    ax.set_xticks([0.4, 1.4, 2.4, 3.4])
    ax.set_xticklabels(['1', '2', '3', '4'])

    pylab.legend(loc="upper right")
    pylab.savefig(filename + ".pdf")
    pylab.savefig(filename + ".eps")
    pylab.savefig(filename + ".svg")

def __work():
    print "Reading", filename

    o = iris(filename)

    pickle.dump(o, open(filename + ".bin", "w"))

    print "Written", filename + ".bin"

    data, labels = o
    display(data, labels)

    if (noshow == 0):
        print "Displaying... (close pylab to continue)"
        pylab.show()

# Comment to do nothing
__work()
