# donut.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import pickle
import sys
from matplotlib import pylab

filename = "teddy_toy_data"

# You can mess with those:
dim_x = 4
dim_y = 4
clust_dist = 20.
clust_size = 200
clust_stddev = 3.

noshow = 0  # Do not show the pylab window

# Argument processing
if sys.argv[len(sys.argv) - 1] == '--noshow':
    noshow = 1

if len(sys.argv) >= 2:
    filename = sys.argv[1]

if len(sys.argv) >= 7:
    dim_x = int(sys.argv[2])
    dim_y = int(sys.argv[3])
    clust_dist = float(sys.argv[4])
    clust_size = int(sys.argv[5])
    clust_stddev = float(sys.argv[6])

# Teddy toy creation
def gen_grid_normal_dataset(dim_tuple, clust_dist, clust_size, clust_stddev):
    data = numpy.empty((clust_size * dim_tuple[0] * dim_tuple[1], 2))
    dx = dim_tuple[0]
    dy = dim_tuple[1]

    n = 0
    for x in range(dx):
        for y in range(dy):
            cx = (x + 1) * clust_dist
            cy = (y + 1) * clust_dist
            px = numpy.random.normal(cx, clust_stddev, clust_size)
            py = numpy.random.normal(cy, clust_stddev, clust_size)
            i = 0
            while i < len(px):
                data[n] = [px[i], py[i]]
                i += 1
                n += 1

    numpy.random.shuffle(data)

    return data

def display(data):
    i = 0
    pylab.figure()
    for d in data:
        pylab.scatter([d[0]], [d[1]], c="r", marker='o')

    pylab.suptitle("Generated teddy toy\ndim=(" + str(dim_x) + ", " + str(dim_y) + ") dist=" + str(clust_dist) + " size=" + str(clust_size) + " stddev=" + str(clust_stddev))

    pylab.savefig(filename + ".pdf")
    pylab.savefig(filename + ".eps")
    pylab.savefig(filename + ".svg")

def __work():
    print "Generating " + filename + " [dim=(" + str(dim_x) + ", " + str(dim_y) + ") dist=" + str(clust_dist) + " size=" + str(clust_size) + " stddev=" + str(clust_stddev) + "]"

    o = gen_grid_normal_dataset((dim_x, dim_y), clust_dist,
                                clust_size, clust_stddev)
    pickle.dump(o, open(filename + ".bin", "w"))

    print "Generating figures"
    display(o)

    if (noshow == 0):
        print "Displaying... (close pylab to continue)"
        pylab.show()

# Comment to do nothing
__work()
