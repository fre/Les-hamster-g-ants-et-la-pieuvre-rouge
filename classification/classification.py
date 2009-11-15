# classification.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import copy
import sys
import pickle

from matplotlib import pylab

# Own stuff
import knn
import knn_nbf
import knn_vdm
import knn_mdv
import kfcv
import bayes
import bayes_ndist
import roc
import id3
import svm
import discretize

filename = "mushroom_data"
plabel = 'e'

# You can mess with those:
slices = 30   # Slices in decision function grid
kcvs = 10     # Slices in kfcv

data_size = 0 # Truncate the data size (0 to disable)

# Also, see the bottom of the file to test your own functions

noshow = 0  # Do not show the pylab window

if sys.argv[len(sys.argv) - 1] == '--noshow':
    noshow = 1

if len(sys.argv) >= 2:
    filename = sys.argv[1]
    plabel = ''

if len(sys.argv) >= 3:
    plabel = sys.argv[2]

if len(sys.argv) >= 4:
    data_size = int(sys.argv[3])

def __kfcrossval(data, labels, names, knns, suffix, results):
    print "  - computing k-fold cross validation"

    if results != None:
        avg, dev = kfcv.kfcrossval(kcvs, knns, data, labels, results)
    else:
        avg, dev = kfcv.kfcrossval(kcvs, knns, data, labels)

    print "    * recognition rates:", avg

    print "  - generating figures"

    ind = numpy.arange(len(avg))
    width = 0.8
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, avg, width,
                    color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'][0:len(ind)],
                    yerr=dev, ecolor='k')

    ax.set_ylabel('Classification rate')
    title = 'Average correct classification rate per classifier'
    title += '\n' + filename
    if data_size != 0:
        title += ' [Data size = ' + str(data_size) + ']'
    ax.set_title(title)
    ax.set_xticks(ind + 0.4)
    ax.set_xticklabels(names)
    for t in ax.get_xticklabels():
     t.set_fontsize(10)

    ax.set_ylim((0.0, 1.0))

    pylab.savefig("images/" + filename + suffix + "_kfcrossval" + ".pdf")
    pylab.savefig("images/" + filename + suffix + "_kfcrossval" + ".eps")
    pylab.savefig("images/" + filename + suffix + "_kfcrossval" + ".svg")

def __roc(roc_results, labels, plabel, names, suffix):
    print "  - computing ROC graph and curves"
    curves, graph = roc.roc(names, roc_results, labels, plabel)

    print "  - generating figures"

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    cu = []
    for cx, cy in curves:
        cu.append(ax.plot(cx, cy))

    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.0))
    fig.legend(cu, names, 'lower right')

    ax.set_ylabel('True positive rate')
    ax.set_xlabel('False positive rate')

    title = 'ROC curve per classifier'
    title += '\n' + filename
    if data_size != 0:
        title += ' [Data size = ' + str(data_size) + ']'
    ax.set_title(title)

    pylab.savefig("images/" + filename + suffix + "_roc" + ".pdf")
    pylab.savefig("images/" + filename + suffix + "_roc" + ".eps")
    pylab.savefig("images/" + filename + suffix + "_roc" + ".svg")

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    cu = []
    for cx, cy in graph:
        cu.append(ax.plot([cx], [cy], marker='o'))

    ax.set_xlim((0.0, 1.0))
    ax.set_ylim((0.0, 1.0))
    fig.legend(cu, names, 'lower right')

    ax.set_ylabel('True positive rate')
    ax.set_xlabel('False positive rate')
    title = 'ROC graph per classifier'
    title += '\n' + filename
    if data_size != 0:
        title += ' [Data size = ' + str(data_size) + ']'
    ax.set_title(title)

    pylab.savefig("images/" + filename + suffix + "_roc_graph" + ".pdf")
    pylab.savefig("images/" + filename + suffix + "_roc_graph" + ".eps")
    pylab.savefig("images/" + filename + suffix + "_roc_graph" + ".svg")

def test_kfcv_roc(in_data, labels, names, cls, suffix, plabel):
    roc_results = []
    suffix = suffix + "_" + str(data_size)

    print "K-fold cross validation, cls =", names, " slices =", kcvs
    __kfcrossval(in_data, labels, names, cls, suffix, roc_results)

    if plabel != '':
        print "ROC generation, cls =", names, " positive label =", plabel
        __roc(roc_results, labels, plabel, names, suffix)

def __work():

    # Main
    print "####  Test for", filename, " ####"
    data, labels = pickle.load(open(filename + ".bin", "r"))

    if data_size > 0:
        print "Reducing data size to", data_size
        data = data[0:data_size]
        labels = labels[0:data_size]

    cls = [id3.ID3(0, id3.gain),
           svm.SVM(),
           knn.KNN(5, cache=knn.cache_5),
           bayes_ndist.BAYES_NDIST(),
           bayes.BAYES()]
    names = ["ID3 Gain",
             "SVM",
             "KNN(5)\nNaive",
             "Bayes\nNormal distribution",
             "Naive\nBayes"]

    in_data = [data,
               data,
               data,
               data,
               data]

    roc_results = []

    test_kfcv_roc(in_data, labels, names, cls, "_2_class_discrete", plabel)

    cls = [id3.ID3(0, id3.gain),
           id3.ID3(0, id3.gainratio),
           id3.ID3(0, id3.gini),
           knn.KNN(5, cache=knn.cache_5),
           bayes_ndist.BAYES_NDIST(),
           bayes.BAYES()]
    names = ["ID3 Gain",
             "ID3 Gain Ratio",
             "ID3 Gini",
             "KNN(5)\nNaive",
             "Bayes\nNormal distribution",
             "Naive\nBayes"]

    in_data = [data,
               data,
               data,
               data,
               data,
               data]

    roc_results = []

    test_kfcv_roc(in_data, labels, names, cls, "_id3", plabel)

    cls = [id3.ID3(0, id3.gain),
           id3.ID3(0, id3.gain),
           id3.ID3(0, id3.gainratio),
           id3.ID3(0, id3.gainratio),
           id3.ID3(0, id3.gini),
           id3.ID3(0, id3.gini)]
    names = ["ID3 Gain\nEWD",
             "ID3 Gain\nEFD",
             "ID3 Gain ratio\nEWD",
             "ID3 Gain ratio\nEFD",
             "ID3 Gini\nEWD",
             "ID3 Gini\nEFD"]

    in_data = [discretize.Discretize().ewd(data),
               discretize.Discretize().efd(data),
               discretize.Discretize().ewd(data),
               discretize.Discretize().efd(data),
               discretize.Discretize().ewd(data),
               discretize.Discretize().efd(data)]

    roc_results = []

    test_kfcv_roc(in_data, labels, names, cls, "_id3_discrete", plabel)

    cls = [knn.KNN(1, cache=knn.cache_100),
           knn.KNN(3, cache=knn.cache_100),
           knn.KNN(5, cache=knn.cache_100),
           knn.KNN(10, cache=knn.cache_100),
           knn.KNN(50, cache=knn.cache_100)]
    names = ["KNN(1)",
             "KNN(3)",
             "KNN(5)",
             "KNN(10)",
             "KNN(50)"]
    in_data = [data,
               data,
               data,
               data,
               data]

    test_kfcv_roc(in_data, labels, names, cls, "_knns", plabel)


    cls = [knn.KNN(5, cache=knn.cache_5),
           knn_nbf.KNN_NBF(5, cache=knn.cache_5),
           knn_vdm.KNN_VDM(5, cache=knn.cache_5),
           knn_mdv.KNN_MDV(5, cache=knn.cache_5),
           bayes.BAYES()]
    names = ["KNN(5)\nNaive",
             "KNN(5)\nNBF",
             "KNN(5)\nVDM",
             "KNN(5)\nMDV",
             "Naive\nBayes"]
    in_data = [data,
               data,
               data,
               data,
               data]

    roc_results = []

    test_kfcv_roc(in_data, labels, names, cls, "_cnt", plabel)

    cls = [knn.KNN(5, cache=knn.cache_5),
           bayes_ndist.BAYES_NDIST(),
           bayes.BAYES()]
    names = ["KNN(5)\nNaive",
             "Bayes\nNormal distribution",
             "Naive\nBayes"]
    in_data = [data,
               data,
               data]

    roc_results = []

    test_kfcv_roc(in_data, labels, names, cls, "_pdf", plabel)

    print "All tests completed"

    if (noshow == 0):
        print "Displaying... (close pylab to continue)"
        pylab.show()

# Comment to do nothing
__work()
