# bayes.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import copy
import sys

#
# BAYES class definition
#
class BAYES(object):
    def __init__(self):
        self.data = []

    def train(self, data, labels):
        self.data = data
        self.labels = labels
        ld = {} # P(Label)
        dd = [] # P(X=x|Label)

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

    def process(self, data):
        result = []
        i = 0
        for d in data:
            ls = self.ld.keys()
            rs = []
            rsum = 0.
            for l in ls:
                r = self.ld[l]
                n = 0
                for v in d:
                    if v in self.dd[n]:
                        di, c = self.dd[n][v]
                        if l in di:
                            r *= di[l]
                        else:
                            r = 0
                            break
                    n += 1
                rsum += r
                rs.append((r, l))

            rr, rl = max(rs)
            if rr != 0:
                rr = rr / rsum
            result.append((rl, rr))

            i += 1

        return result
