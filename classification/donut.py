# donut.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import pickle
import sys
from matplotlib import pylab

filename = "donut_data"

# You can mess with those:
r1 = 3.5    # Inner radius
r2 = 8.5    # Outer radius
max_xy = 10 # Max spread

samples = 1000 # Number of samples
noise = 0.1    # Noise factor

noshow = 0  # Do not show the pylab window

# Argument processing
if sys.argv[len(sys.argv) - 1] == '--noshow':
    noshow = 1

if len(sys.argv) >= 2:
    filename = sys.argv[1]

if len(sys.argv) >= 4:
    samples = int(sys.argv[2])
    noise = float(sys.argv[3])

# Donut creation
def donut_toy(n, noise):
    data = numpy.empty((n, 2))
    labels = []
    for i in range(n):
        x = (numpy.random.random_sample() - 0.5) * max_xy * 2
        y = (numpy.random.random_sample() - 0.5) * max_xy * 2
        data[i] = [x, y]
        r = math.sqrt(x * x + y * y)
        noise_test = numpy.random.random_sample()
        if ((noise_test < noise) and (r < r1 or r > r2))             \
           or (not (noise_test < noise) and not (r < r1 or r > r2)):
            labels.append("inside")
        else:
            labels.append("outside")
    return data, labels

def display(data, labels):
    i = 0
    pylab.figure()
    for d in data:
        if labels[i] == "inside":
            pylab.scatter([d[0]], [d[1]], c="r", marker='^')
        else:
            pylab.scatter([d[0]], [d[1]], c="b", marker='o')
        i = i + 1
    pylab.suptitle("Generated donut (samples=" + str(samples) + " noise=" + str(noise) + ")")
    pylab.axis([-max_xy - 1, max_xy + 1, -max_xy - 1, max_xy + 1])
    pylab.savefig(filename + ".pdf")
    pylab.savefig(filename + ".eps")
    pylab.savefig(filename + ".svg")

print "Generating " + filename + ", samples =", samples, " noise =", noise

def __work():
    o = donut_toy(samples, noise)
    pickle.dump(o, open(filename + ".bin", "w"))
    data, labels = o
    display(data, labels)

    if (noshow == 0):
        print "Displaying... (close pylab to continue)"
        pylab.show()

# Comment to do nothing
__work()
