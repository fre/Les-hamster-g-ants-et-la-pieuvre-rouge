# kfcv.py - TP MLEA d-hall_f
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
# K-fold cross validation
#
# Store all classification results in results, if provided
#
def kfcrossval(k, l_classifiers, data, labels, results=None):
    avg = [0.] * len(l_classifiers)
    dev = [0.] * len(l_classifiers)
    zerovals = [0.] * k
    vals = [zerovals[:] for i in range(len(l_classifiers))]

    if results != None:
        for c in l_classifiers:
            results.append([])

    for i in range(k):
        # divide data
        d = data[0]
        mini = len(d) * i / k;
        maxi = len(d) * (i + 1) / k
        if len(data) == 1:
            training_data = numpy.concatenate((d[0:mini], d[maxi:]))
            training_labels = numpy.concatenate((labels[0:mini],
                                                 labels[maxi:]))
            test_data = d[mini:maxi]
            test_labels = labels[mini:maxi]

        # compute values
        n = 0
        for c in l_classifiers:
            if len(data) != 1:
                d = data[n]
                training_data = numpy.concatenate((d[0:mini], d[maxi:]))
                training_labels = numpy.concatenate((labels[0:mini],

                                                     labels[maxi:]))
                test_data = d[mini:maxi]
                test_labels = labels[mini:maxi]

            c.train(training_data, training_labels)
            res_labels = c.process(test_data)
            if results != None:
                for r in res_labels:
                    results[n].append(r)
            for j in range(len(res_labels)):
                if res_labels[j][0] == test_labels[j]:
                    vals[n][i] += 1.
            vals[n][i] = vals[n][i] / len(res_labels)
            n += 1

    for n in range(len(l_classifiers)):
        avg[n] = numpy.mean(vals[n])
        dev[n] = numpy.std(vals[n])
    return avg, dev
