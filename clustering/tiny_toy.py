# donut.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import pickle
import sys
from matplotlib import pylab

filename = "tiny_toy_data"

noshow = 1

# Argument processing
if sys.argv[len(sys.argv) - 1] == '--noshow':
    noshow = 1

if len(sys.argv) >= 2:
    filename = sys.argv[1]

# tiny toy creation
def tiny_toy():
    return numpy.array([[1.], [2.], [7.], [8.], [9.], [10.]])

print "Generating " + filename

def __work():
    o = tiny_toy()
    pickle.dump(o, open(filename + ".bin", "w"))
    print "Written", filename + ".bin"

    if (noshow == 0):
        print "Displaying... (close pylab to continue)"
        pylab.show()

# Comment to do nothing
__work()
