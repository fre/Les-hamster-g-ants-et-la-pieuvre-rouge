# k_means.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import copy
import sys

def stddistance(x, y):
    d = x - y
    return numpy.dot(d, d)

def stdcenter(d):
    pt = numpy.sum(d, 0)
    pt /= len(d)
    return pt

#
# class KMEANS
#
class KMEANS(object):
    def __init__(self, K=5, distance=stddistance, center=stdcenter):
        self.k = K
        self.distance = distance
        self.center = center

    def process(self, data):
        centers = numpy.empty([self.k, data.shape[1]])
        if data.shape[0] < self.k:
            print "    *** Not enough data points, reducing K to " + str(data.shape[0]) + " ***"
            self.k = data.shape[0]

        n = 0
        for i in numpy.random.random_integers(0, data.shape[0] - 1, self.k):
            j = i
            while data[j] in centers[0:n]:
                j = (j + 1) % data.shape[0]
            centers[n] = data[j]
            n += 1

        # Iterations
        results = []
        done = False
        iterations = 0
        while not done:
            clusters = []
            distances = []
            sdistances = []
            for i in range(self.k):
                clusters.append([])
            id = 0
            for d in data:
                dst = []
                i = 0
                for k in centers:
                    dst.append((self.distance(d, k), i))
                    i += 1
                dst.sort()
                clusters[dst[0][1]].append(d)
                di = dst[0][0]
                distances.append(di)
                sdistances.append(di * di)
                id += 1
            ct = numpy.empty([self.k, data.shape[1]])
            for i in range(self.k):
                ct[i] = (self.center(clusters[i]))
            if numpy.all(ct == centers):
                done = True
            results.append((copy.copy(centers), copy.copy(clusters),
                            sum(distances), sum(sdistances), iterations))
            centers = ct
            iterations += 1

        return results
