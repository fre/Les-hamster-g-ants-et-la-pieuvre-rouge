# knn.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import copy
import sys

#
# Distances
#
def stddistance(x, y, t):
    d = x - y
    return numpy.dot(d, d)

def distance_sum(x, y, t):
    d = x - y
    return d.__imod__(2).sum()

# actually 40% slower than std
def distance_fast(x, y, t):
    ds = 2
    d = x - y
    res = 0
    for i in range(ds):
        # divide data
        mini = d.shape[0] * i / ds;
        maxi = d.shape[0] * (i + 1) / ds
        res += numpy.dot(d[mini:maxi], d[mini:maxi])
        if res > t:
            return 10e300
    return res;

def distance_manhattan(x, y, t):
    d = x - y
    r = 0.
    for v in d:
        r += abs(v)
    return r

def distance_max(x, y, t):
    d = x - y
    for i in range(d.shape[0]):
        d[i] = abs(d[i])
    return numpy.max(d)

#
# Weights
#
def stdweight(d):
    return 1.

def weight_c1(d):
    return d

def weight_c2(d):
    return 1. / (1. + math.sqrt(d))

def weight_c3(d):
    return 1. + math.exp(-d)


#
# Cache for KNN
#
# Cache nearest neighbors on a data set.
#
class KNN_CACHE(object):
    def __init__(self, K=5, distance=stddistance):
        self.k = K
        self.distance = distance
        self.cached = False
        self.data = None
        self.labels = None

    def train(self, data, labels):
        if numpy.any(data != self.data) or numpy.any(labels != self.labels):
            self.data = data
            self.labels = labels
            self.cached = False

    def __nn(self, v):
        nn = [(10e300, 'null')] * self.k
        maxd = 10e300

        for i in range(self.data.shape[0]): # Perf hog
            d = self.distance(v, self.data[i], maxd)
            j = 0
            while j < self.k:
                if nn[j][0] - d <= 0:
                    break
                j += 1

            if j != 0:
                nni = [(d, self.labels[i])]
                if j > 1:
                    nni = nn[1:j] + nni
                if j < self.k:
                    nni = nni + nn[j:self.k]
                nn = nni
                maxd = nn[0][0]

        return nn

    def __nn_nosort(self, v):
        nn = []
        for i in range(self.data.shape[0]):
            d = self.distance(v, self.data[i], 10e300)
            nn.append((d, self.labels[i]))
        nn.sort(reverse=True)
        nn = nn[len(nn) - self.k:len(nn)]
        return nn

    def nn(self, i, k):
        if (self.cached):
            l = len(self.nns[i])
            return self.nns[i][l - k:l]

    def process(self, data):
        if (not self.cached or numpy.any(self.cdata != data)):
            self.cdata = data
            self.cached = True
            self.nns = []
            if self.k > 50:
                for v in data:
                    self.nns.append(self.__nn_nosort(v))
            else:
                for v in data:
                    self.nns.append(self.__nn(v))
            return

cache_1 = KNN_CACHE(1)
cache_5 = KNN_CACHE(5)
cache_10 = KNN_CACHE(10)
cache_50 = KNN_CACHE(50)
cache_100 = KNN_CACHE(100)
cache_500 = KNN_CACHE(500)
cache_1000 = KNN_CACHE(1000)
cache_5000 = KNN_CACHE(5000)

#
# KNN class definition
#
class KNN(object):
    def __init__(self, K=5, distance=stddistance,
                 weight=stdweight, cache=None):
        self.k = K
        self.distance = distance
        self.weight = weight
        self.cached = False
        if cache:
            self.cached = True
            self.cache = cache

    # We could get a big performance gain by partitioning
    # the space (BSP...), and/or reducing the amount of
    # data used (cleaning...).
    def train(self, data, labels):
        self.data = data
        self.labels = labels
        if self.cached:
            self.cache.train(data, labels)

    def __nn(self, v):
        nn = [(10e300, 'null')] * self.k
        maxd = 10e300

        for i in range(self.data.shape[0]): # Perf hog
            d = self.distance(v, self.data[i], maxd)
            j = 0
            while j < self.k:
                if nn[j][0] - d <= 0:
                    break
                j += 1

            if j != 0:
                nni = [(d, self.labels[i])]
                if j > 1:
                    nni = nn[1:j] + nni
                if j < self.k:
                    nni = nni + nn[j:self.k]
                nn = nni
                maxd = nn[0][0]

        return nn

    def process(self, data):
        if self.cached:
            self.cache.process(data)
        result = []
        i = 0
        for v in data:
            if self.cached:
                nn = self.cache.nn(i, self.k)
            else:
                nn = self.__nn(v)

            labels = {}
            for d, l in nn:
                if l in labels:
                    labels[l] += self.weight(d)
                else:
                    labels[l] = self.weight(d)
            maxc = 0
            label = ""
            for l, c in labels.iteritems():
                if c >= maxc:
                    maxc = c
                    label = l
            result.append((label, (maxc * 1.) / self.k))

            i += 1

        return result

    def decision(self, data):
        if self.cached:
            self.cache.process(data)
        labels = {}
        for l in self.labels:
            labels[l] = 0
        result = numpy.empty((len(data), len(labels.keys())))
        keys = sorted(labels.keys())
        id = 0
        for d in data:
            ls = copy.copy(labels) # Assignment is made by ref
            if self.cached:
                for d, l in self.cache.nn(id, self.k):
                    ls[l] += self.weight(d)
            else:
                for d, l in self.__nn(d):
                    ls[l] += self.weight(d)

            vals = numpy.empty(len(labels))
            i = 0
            for k in keys:
                result[id][i] = ls[k]
                i += 1
            id += 1
        return result
