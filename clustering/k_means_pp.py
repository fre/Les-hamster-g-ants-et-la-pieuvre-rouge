# k_means_pp.py - TP MLEA d-hall_f
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

def choose_std(data, dd, dds):
    # p is 'normalized' with the sum of squared distances
    p = numpy.random.random_sample() * dds
    csum = 0.

    for i, d in enumerate(data):
        csum += dd[i]
        if p < csum:
            return i

def choose_max(data, dd, dds):
    cmax = 0.
    c = 0
    for i, d in enumerate(data):
        if dd[i] > cmax:
            cmax = dd[i]
            c = i
    return c

#
# class KMEANSPP
#
class KMEANSPP(object):
    def __init__(self, K=5, centerchoice=choose_std,
                 distance=stddistance, center=stdcenter):
        self.k = K
        self.distance = distance
        self.center = center
        self.choosecenter=centerchoice

    def process(self, data):
        centers = numpy.empty([self.k, data.shape[1]])
        if data.shape[0] < self.k:
            print "    *** Not enough data points, reducing K to " + str(data.shape[0]) + " ***"
            self.k = data.shape[0]

        # Choosing centers
        dd = numpy.empty(data.shape[0])
        centers[0] = data[numpy.random.random_integers(0,
                                                       data.shape[0] - 1, 1)[0]]
        i = 0
        for d in data:
            dst = self.distance(centers[0], d)
            dd[i] = dst * dst
            i += 1
        dds = numpy.sum(dd)

        n = 1
        while n < self.k:
            c = self.choosecenter(data, dd, dds)

            centers[n] = data[c]
            # Recompute shortest distances
            for i, d in enumerate(data):
                dst = self.distance(centers[n], d)
                dst = dst * dst
                if dst < dd[i]:
                    dd[i] = dst
            dds = numpy.sum(dd)
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
