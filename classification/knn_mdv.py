# knn_mdv.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import copy
import sys
import knn

#
# KNN_MDV class definition
#
class KNN_MDV(object):
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
                if l in dd[n]:
                    di, c = dd[n][l]
                    if v in di:
                        di[v] += 1.
                    else:
                        di[v] = 1.
                    dd[n][l] = (di, c + 1.)
                else:
                    dd[n][l] = ({v: 1.}, 1.)
                n += 1
            nd += 1

        for l in ld.keys():
            ld[l] = ld[l] / data.shape[0]
        for d in dd:
            for l in d.keys():
                di, c = d[l]
                for v in di.keys():
                    di[v] = di[v] / c

        self.ld = ld
        self.dd = dd
        self.ldkeys = self.ld.keys()

        ndata = []
        for d in data:
            r = []
            i = 0
            for v in d:
                for l in self.ldkeys:
                    px = 0.
                    if l in self.dd[i]:
                        di, c = self.dd[i][l]
                    if d[i] in di:
                        px = di[d[i]]
                i += 1
                r.append(px)
            ndata.append(r)
        self.knn.train(numpy.array(ndata), labels)

    def process(self, data):
        ndata = []
        for d in data:
            r = []
            i = 0
            for v in d:
                for l in self.ldkeys:
                    px = 0.
                    if l in self.dd[i]:
                        di, c = self.dd[i][l]
                    if d[i] in di:
                        px = di[d[i]]
                i += 1
                r.append(px)
            ndata.append(r)
        return self.knn.process(numpy.array(ndata))
