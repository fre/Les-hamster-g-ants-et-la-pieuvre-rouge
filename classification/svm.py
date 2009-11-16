# svm.py - TP MLEA
#
# See comments within the source code.

import numpy
import pickle
import math
import copy
import sys
import cvxopt

from matplotlib import pylab
from cvxopt.solvers import qp
from cvxopt.base import matrix

max_xy = 10 # Max spread

if len(sys.argv) >= 2:
    filename = sys.argv[1]

def linear_k(x, y, p_value):
    val = numpy.sum(numpy.dot(x,y), dtype=float) + 1
    return val

def polynomial_k(x, y, p_value):
    val = numpy.power((numpy.sum(numpy.dot(x,y), dtype=float) + 1), int(p_value))
    return val

def rbf_k(x, y, p_value):
    m = x - y
    val = numpy.exp(-1 * numpy.dot(m, m) / float(p_value))
    return val

def norm_k(x, y):
    x = numpy.array(x)
    y = numpy.array(y)
    val = kernel(x, y) / numpy.sqrt(kernel(x, x) * kernel(y, y))
    return val

test = 0
p_value = 1
kernel = rbf_k
if sys.argv[-1] == '--test':
    test = 1
    if len(sys.argv) >= 5:
        if sys.argv[-3] == '--poly_k':
            kernel = polynomial_k
            p_value = sys.argv[-2]
        if sys.argv[-3] == '--rbf_k':
            kernel = rbf_k
            p_value = sys.argv[-2]
    if len(sys.argv) >= 4:
        if sys.argv[-2] == '--linear_k':
            kernel = linear_k


class SVM(object):
    def __init__(self, kernel = rbf_k, p_value = 0.05):
        self.kernel = kernel
        self.alpha = []
        self.bias = 0.
        self.weight = 0.
        self.p_value = p_value

    def train(self, data, labels):
        self.data = data
        l_map = {}
        self.labels = []
        i = -1.
        for l in labels:
            if l in l_map:
                self.labels.append(l_map[l])
            else:
                if i == 3:
                    print "Training error: dataset contains more"\
                        "than two classes."
                l_map[l] = i
                self.labels.append(i)
                i = i + 2
        self.label_map = {-1.: '', 1.: ''}
        for l, v in l_map.iteritems():
            self.label_map[v] = l
        self.alpha = self.lagrange_coeffs(data, self.labels)
        self.bias = self.bias_get(data, self.labels)

    # Get weight vector
    def __weight(self, data, labels):
        w = 0.
        for i in xrange(0, len(self.alpha)):
            w += (self.alpha[i][0] * labels[self.alpha[i][1]]) * data[self.alpha[i][1]]
        return w

    # Get value of the decision
    def decision(self, data):
        deci = []
        for d in data:
            w = 0.
            for i in xrange(0, len(self.alpha)):
              w += self.alpha[i][0] * self.labels[self.alpha[i][1]] \
                    * self.kernel(self.data[self.alpha[i][1]], d, self.p_value)
            deci.append(w)
        return deci

    # Get new label
    def process(self, data):
        deci = self.decision(data)
        new_lab = []

        for d in deci:
            if d >= 0:
                new_lab.append((self.label_map[1], min(1., 1. - d)))
            else:
                new_lab.append((self.label_map[-1], max(0., d - 1.)))
        return new_lab

    # get bias of svm (page 12-14)
    def bias_get(self, data, labels):
        self.weight = self.__weight(data, labels)

        b = 0.
        for i in xrange(0, len(self.alpha)):
            b += (labels[self.alpha[i][1]] - norm_k(self.weight, data[self.alpha[i][1]]))

        return b / len(self.alpha)

    # Get the matrice P for the solver
    def __get_quad(self, data, n, l):
        p = (n,n)
        P = numpy.zeros(p)
        for i in xrange(0, n):
            for j in xrange(i, n):
                # (see Annexe page 3 formula 8)
                val = l[i] * l[j] * self.kernel(data[i], data[j], self.p_value)
                P[i, j] = val
                P[j, i] = val
        return P

    # get lagrange coefficient (alpha vector)
    # use annexe to understand
    def lagrange_coeffs(self, data, labels):
        n = len(labels)
        q = numpy.dot(numpy.ones(n), -1)
        h = numpy.zeros(n)
        G = numpy.dot(numpy.eye(n), -1)
        P = self.__get_quad(data, n, labels)
        # Constraint value for Sum(alpha_i*label_i)=0
        b = numpy.zeros(1)
        # Constraite propriety Sum(alpha_i*label_i)=0
        A = matrix(labels, (1, n))
        r = qp(matrix(P), matrix(q), matrix(G), matrix(h), A, matrix(b))
        # Get alpha vector
        al = numpy.array(r['x'])
        al = al.round()
        # Delete 0 value
        alpha = [(int(al[i]), i)  for i in xrange(0, len(al)) if al[i] <> 0]
        return alpha

    def print_2Ddecision(self, bounds, print_sv=True, print_non_sv=False):
        def get_grid(step):
            step *= 10;
            grid = [];
            for i in xrange (-100., 100., step):
                for j in xrange (-100., 100., step):
                    grid.append([i/10., j/10.]);
            return grid

        step = .5
        data = get_grid(step)
        deci = self.decision(data)

        pylab.figure()

        # number of row and line
        row = 20 / step;
        # Transform the array into matrix
        mat = numpy.reshape(numpy.array(deci), (row, row));

        im = pylab.imshow(mat, extent=[bounds[0], bounds[1], bounds[2], bounds[3]])

        pylab.colorbar(im)
        pylab.axis("off")

        if (print_sv):
            X = []
            Y = []
            for a in self.alpha:
                X.append(self.data[a[1]][0])
                Y.append(self.data[a[1]][1])
            pylab.scatter(X, Y, s=100, c="r", marker='o')

        if (kernel == linear_k):
            pylab.savefig(filename + "_linear_s" + str(len(self.alpha)) + ".pdf")
            pylab.savefig(filename + "_linear_s" + str(len(self.alpha)) + ".eps")
            pylab.savefig(filename + "_linear_s" + str(len(self.alpha)) + ".svg")
        elif (kernel == polynomial_k):
            pylab.savefig(filename + "_p" + p_value + "_s" + str(len(self.alpha)) + ".pdf")
            pylab.savefig(filename + "_p" + p_value + "_s" + str(len(self.alpha)) + ".eps")
            pylab.savefig(filename + "_p" + p_value + "_s" + str(len(self.alpha)) + ".svg")
        elif (kernel == rbf_k):
            pylab.savefig(filename + "_g" + p_value + "_s" + str(len(self.alpha)) + ".pdf")
            pylab.savefig(filename + "_g" + p_value + "_s" + str(len(self.alpha)) + ".eps")
            pylab.savefig(filename + "_g" + p_value + "_s" + str(len(self.alpha)) + ".svg")

        return

def __test():
    print "Testing SVM..."
    data, labels = pickle.load(open(filename + ".bin", "r"))

# Tiny probleme
#     d = numpy.array([[1], [2], [3]])
#     l = [-1.0, 1.0, 1.0]

    svm = SVM(kernel, p_value);
    svm.train(data, labels)
#     lab = svm.process(data)

    svm.print_2Ddecision([-10, 10, -10, 10])
if test:
    __test()
