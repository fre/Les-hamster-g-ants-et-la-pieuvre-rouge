# roc.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import copy
import sys
import pickle
import knn

from matplotlib import pylab

#
# ROC graph & curve
#
# Only works for 2 labels
#
def roc(cnames, results, labels, plabel):
    res = []
    cur = []
    gra = []
    TPn = []
    TNn = []
    FPn = []
    FNn = []

    for c in cnames:
        res.append([])
        TPn.append(0.)
        TNn.append(0.)
        FPn.append(0.)
        FNn.append(0.)

    n = 0
    for c in results:
        i = 0
        for l, r in c:
            if l == plabel:
                res[n].append((r, l, labels[i]))
                if l == labels[i]:
                    TPn[n] += 1.
                else:
                    FPn[n] += 1.
            else:
                res[n].append((1. - r, l, labels[i]))
                if l == labels[i]:
                    TNn[n] += 1.
                else:
                    FNn[n] += 1.
            i += 1
        res[n].sort(reverse=True)
        n += 1

    i = 0
    for L in res:
        FP = 0.
        TP = 0.
        P = TPn[i] + FNn[i]
        N = TNn[i] + FPn[i]
        Rx = []
        Ry = []
        fp = -10e300

        for f, l, tl in L:
            if f != fp:
                Rx.append(FP/N)
                Ry.append(TP/P)
                fp = f
            if l == tl and l == plabel:
                TP += 1.
            if l != tl and l == plabel:
                FP += 1.

        Rx.append(1)
        Ry.append(1)

        cur.append((Rx, Ry))
        gra.append((FPn[i] / N, TPn[i] / P))
        i += 1

    return (cur, gra)
