# bayes_ndist.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import copy
import sys

def stdpdf(v, mean, std):
    if (std != 0):
        res = 1. / std / math.sqrt(2. * math.pi)
        res *= math.exp(-0.5 * math.pow((v - mean) / std, 2))
        return res
    else:
        if v == mean:
            return 1.
        else:
            return 0.

#
# BAYES_NDIST class definition
#
class BAYES_NDIST(object):
    def __init__(self):
        self.data = []
        self.pdf = stdpdf

    def train(self, data, labels):
        self.data = data
        self.labels = labels
        ld = {} # [P(Label), mean, std, values]

        for nd, d in enumerate(data):
            l = labels[nd]
            if l in ld:
                ld[l][0] += 1.
                for n, v in enumerate(d):
                    ld[l][3][n].append(v)
            else:
                ld[l] = [1., [], [], []]
                for v in d:
                    ld[l][3].append([v])
                    ld[l][2].append(0.)
                    ld[l][1].append(0.)

        for l in ld.keys():
            ld[l][0] = ld[l][0] / data.shape[0]
            for n in range(data.shape[1]):
                ld[l][1][n] = numpy.mean(ld[l][3][n])
                ld[l][2][n] = numpy.std(ld[l][3][n])

        self.ld = ld

    def process(self, data):
        result = []
        i = 0
        for d in data:
            ls = self.ld.keys()
            rs = []
            rsum = 0.
            for l in ls:
                ldd = self.ld[l]
                r = ldd[0]
                for n, v in enumerate(d):
                    r *= self.pdf(v, ldd[1][n], ldd[2][n])
                rsum += r
                rs.append((r, l))

            rr, rl = max(rs)
            if rr != 0:
                rr = rr / rsum
            result.append((rl, rr))

            i += 1

        return result
