# knn_vdm.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import copy
import sys
import knn

#
# KNN_VDM class definition
#
class KNN_VDM(object):
    def __init__(self, K=5, distance=knn.stddistance,
                 weight=knn.stdweight, cache=None):
        self.k = K
        self.knn = knn.KNN(K, distance, weight, cache)

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
                    di, c = dd[n][v]
                    if l in di:
                        di[l] += 1.
                    else:
                        di[l] = 1.
                    dd[n][v] = (di, c + 1.)
                else:
                    dd[n][v] = ({l: 1.}, 1.)
                n += 1
            nd += 1

        for l in ld.keys():
            ld[l] = ld[l] / data.shape[0]
        for d in dd:
            for v in d.keys():
                di, c = d[v]
                for l in di.keys():
                    di[l] = di[l] / c

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
                    if v in self.dd[i]:
                        di, c = self.dd[i][v]
                        if l in di:
                            px = di[l]
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
                    if v in self.dd[i]:
                        di, c = self.dd[i][v]
                        if l in di:
                            px = di[l]
                i += 1
                r.append(px)
            ndata.append(r)
        return self.knn.process(numpy.array(ndata))
