# clustering.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import copy
import sys
import pickle

from matplotlib import pylab

# Own stuff
import k_means
import k_means_pp

filename = "teddy_toy_data"

# You can mess with those:
centroids = 16 # Number of centroids
data_size = 0 # Truncate the input data (0 to disable)
clustering_iters = 10 # Iterations
display_all = False # display all the iterations

# Also, see the bottom of the file to test your own functions

noshow = 0  # Do not show the pylab window

if sys.argv[len(sys.argv) - 1] == '--noshow':
    noshow = 1

if len(sys.argv) >= 2:
    filename = sys.argv[1]

if len(sys.argv) >= 3:
    centroids = int(sys.argv[2])

if len(sys.argv) >= 4:
    data_size = int(sys.argv[3])

if len(sys.argv) >= 5:
    clustering_iters = int(sys.argv[4])

def __display_clusters(centers, clusters, iterations, name, suffix, place):

    fig = pylab.figure()
    ax = fig.add_subplot(111)

    N = len(centers)

    colors_id = numpy.array(range(N), dtype=numpy.float32) / (N - 1)
    cm = pylab.get_cmap('jet')
    colors = cm(colors_id)

    cid = 0
    for c in clusters:
        for p in c:
            x = p[0]
            if p.shape[0] > 1:
                y = p[1]
            else:
                y = 0
            ax.scatter([x], [y], c=colors[cid], marker='o')
        cid += 1

    cid = 0
    for c in centers:
        x = c[0]
        if c.shape[0] > 1:
            y = c[1]
        else:
            y = 0
        ax.scatter([x], [y], c=colors[cid], s=200, marker='^')
        cid += 1

    #ax.set_ylabel('True positive rate')
    #ax.set_xlabel('False positive rate')

    title = name + ' clustering'
    title += '\n' + filename
    title += ' [' + place + '] '
    title += ' [iterations=' + str(iterations)
    if data_size != 0:
        title += ' data size=' + str(data_size)
    title += ']'
    ax.set_title(title)

    ofname = filename + suffix + "_clustering_" + name + "_" + place

    pylab.savefig(ofname + ".pdf")
    pylab.savefig(ofname + ".eps")
    pylab.savefig(ofname + ".svg")

def __display_means(names, dsta, sdsta, itsa, suffix):
    avgd = []
    devd = []
    for dst in dsta:
        avgd.append(numpy.mean(dst))
        devd.append(numpy.std(dst))

    avgsd = []
    devsd = []
    for dst in sdsta:
        avgsd.append(numpy.mean(dst))
        devsd.append(numpy.std(dst))

    avgi = []
    devi = []
    for its in itsa:
        avgi.append(numpy.mean(its))
        devi.append(numpy.std(its))

    ind = numpy.arange(len(names))
    width = 0.8
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, avgd, width,
                    color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'][0:len(ind)],
                    yerr=devd, ecolor='k')

    ax.set_ylabel('distance sum')
    title = 'Average sum of distances to cluster center per classifier'
    title += '\n' + filename
    if data_size != 0:
        title += ' [Data size = ' + str(data_size) + ']'
    ax.set_title(title)
    ax.set_xticks(ind + 0.4)
    ax.set_xticklabels(names)
    for t in ax.get_xticklabels():
        t.set_fontsize(10)

    ofname = filename + suffix + "_clustering_sumdst_" + str(clustering_iters)
    pylab.savefig(ofname + ".pdf")
    pylab.savefig(ofname + ".eps")
    pylab.savefig(ofname + ".svg")

    ind = numpy.arange(len(names))
    width = 0.8
    fig = pylab.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, avgsd, width,
                    color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'][0:len(ind)],
                    yerr=devsd, ecolor='k')

    ax.set_ylabel('squared distance sum')
    title = 'Average sum of squared distances to cluster center per classifier'
    title += '\n' + filename
    if data_size != 0:
        title += ' [Data size = ' + str(data_size) + ']'
    ax.set_title(title)
    ax.set_xticks(ind + 0.4)
    ax.set_xticklabels(names)
    for t in ax.get_xticklabels():
        t.set_fontsize(10)

    ofname = filename + suffix + "_clustering_sumsdst_" + str(clustering_iters)
    pylab.savefig(ofname + ".pdf")
    pylab.savefig(ofname + ".eps")
    pylab.savefig(ofname + ".svg")

    fig = pylab.figure()
    ax = fig.add_subplot(111)
    rects1 = ax.bar(ind, avgi, width,
                    color=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'][0:len(ind)],
                    yerr=devi, ecolor='k')

    ax.set_ylabel('iterations')
    title = 'Average iterations per classifier'
    title += '\n' + filename
    if data_size != 0:
        title += ' [Data size = ' + str(data_size) + ']'
    ax.set_title(title)
    ax.set_xticks(ind + 0.4)
    ax.set_xticklabels(names)
    for t in ax.get_xticklabels():
        t.set_fontsize(10)

    ofname = filename + suffix + "_clustering_sumit_" + str(clustering_iters)
    pylab.savefig(ofname + ".pdf")
    pylab.savefig(ofname + ".eps")
    pylab.savefig(ofname + ".svg")

def __clustering(data, classifier, name, suffix, display, it):
    print "  - computing clusters [" + str(it) + "]"
    r = classifier.process(data)

    if display:
        print "  - generating figures"
        if display_all:
            for i in range(len(r)):
                centers, clusters, cdistances, scdistances, iterations = r[i]
                __display_clusters(centers, clusters, iterations, name, suffix, str(i))
        else:
            centers, clusters, cdistances, scdistances, iterations = r[0]
            __display_clusters(centers, clusters, iterations, name, suffix, "Start")
            centers, clusters, cdistances, scdistances, iterations = r[len(r) - 1]
            __display_clusters(centers, clusters, iterations, name, suffix, "End")

    centers, clusters, cdistances, scdistances, iterations = r[len(r) - 1]

    return centers, clusters, cdistances, scdistances, iterations

def test_clustering(data, cls, names, suffix):
    suffix = suffix + "_" + str(data_size)
    dsta = []
    sdsta = []
    itsa = []

    for i, c in enumerate(cls):
        n = names[i]
        print n + " clustering"
        dst = []
        sdst = []
        its = []
        display = True
        for j in range(clustering_iters):
            ct, cl, cd, sd, it = __clustering(data, c, n, suffix, display, j)
            dst.append(cd)
            sdst.append(sd)
            its.append(it)
            display = False
        dsta.append(dst)
        sdsta.append(sdst)
        itsa.append(its)

    __display_means(names, dsta, sdsta, itsa, suffix)

def __work():

    # Main
    print "####  Test for", filename, " ####"
    data = pickle.load(open(filename + ".bin", "r"))

    if data_size > 0:
        print "Reducing data size to", data_size
        data = data[0:data_size]

    #test_clustering(data, [k_means.KMEANS(16)], ["KMPP"], "_test")

    cls = [k_means.KMEANS(centroids),
           k_means_pp.KMEANSPP(centroids),
           k_means_pp.KMEANSPP(centroids, k_means_pp.choose_max)]
    names = ["K-MEANS(" + str(centroids) + ")",
             "K-MEANSPP(" + str(centroids) + ")",
             "K-MEANSPP(" + str(centroids) + ") (max distance)"]

    test_clustering(data, cls, names, "")

    print "All tests completed"

    if (noshow == 0):
        print "Displaying... (close pylab to continue)"
        pylab.show()

# Comment to do nothing
__work()
