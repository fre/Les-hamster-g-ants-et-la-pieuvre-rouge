# knn_nbf.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import copy
import sys
import knn

#
# KNN_NBF class definition
#
class KNN_NBF(object):
    def __init__(self, K=5, distance=knn.stddistance,
                 weight=knn.stdweight, cache=None):
        self.k = K
        self.knn = knn.KNN(5, distance, weight, cache)

    def train(self, data, labels):
        ld = {}
        dd = []

        for v in data[0]:
            dd.append({})

        nd = 0
        for d in data:
            l = labels[nd]
            if l in ld:
                ld[l] += 1.
            else:
                ld[l] = 1.
            n = 0
            for v in d:
                if v in dd[n]:
                    dd[n][v] += 1.
                else:
                    dd[n][v] = 0.
                n += 1
            nd += 1

        self.ld = ld
        self.dd = dd
        self.ldkeys = self.ld.keys()

        ndata = []
        for d in data:
            r = []
            i = 0
            for v in d:
                for l in self.dd[i].keys():
                    if v == l:
                        r.append(1)
                    else:
                        r.append(0)
                i += 1
            ndata.append(r)
        self.knn.train(numpy.array(ndata), labels)

    def process(self, data):
        ndata = []
        for d in data:
            r = []
            i = 0
            for v in d:
                for l in self.dd[i].keys():
                    if v == l:
                        r.append(1)
                    else:
                        r.append(0)
                i += 1
            ndata.append(r)
        return self.knn.process(numpy.array(ndata))
