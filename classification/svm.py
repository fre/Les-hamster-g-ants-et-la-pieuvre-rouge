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

def linear_k(x, y):
    val = numpy.sum(numpy.dot(x,y), dtype=float) + 1
    return val

def polynomial_k(x, y):
    val = numpy.power((numpy.sum(numpy.dot(x,y), dtype=float) + 1), int(p_value))
    return val

def rbf_k(x, y):
    m = numpy.linalg.norm(numpy.power((x - y), 2))
    val = numpy.exp(-1 * m * float(p_value))
    return val

test = 0
p_value = 1
if sys.argv[-1] == '--test':
    test = 1
if sys.argv[-3] == '--poly_k':
    kernel = polynomial_k
    p_value = sys.argv[-2]
if sys.argv[-3] == '--rbf_k':
    kernel = rbf_k
    p_value = sys.argv[-2]
if sys.argv[-2] == '--linear_k':
    kernel = linear_k


class SVM(object):
    def __init__(self):
        self.kernel = kernel
        self.alpha = []
        self.bias = 0.
        self.weight = 0.


    def train(self, data, labels):
        self.data = data
        self.labels = labels
        self.alpha = self.lagrange_coeffs(data, labels)
        self.bias = self.bias_get(data, labels)

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
                w += self.alpha[i][0] * self.labels[self.alpha[i][1]] * self.kernel(self.data[self.alpha[i][1]], d)
            deci.append(w)
        return deci

    # Get new label
    def process(self, data):
        deci = self.decision(data)
        new_lab = []

        for d in deci:
            if d >= 0:
                new_lab.append(1)
            else:
                new_lab.append(-1)
        return new_lab

    # get bias of svm (page 12-14)
    def bias_get(self, data, labels):
#         self.weight = self.__weight(data, labels)

#         b = 0.
#         for i in xrange(0, len(self.alpha)):
# #             for j in xrange(0, len(self.alpha)):
# #                 k = self.kernel(data[self.alpha[i][1]], data[self.alpha[j][1]])
# #                 print k, self.weight
#             b += (labels[self.alpha[i][1]] - self.kernel(self.weight, data[self.alpha[i][1]]))

#         return b / len(self.alpha)
        return

    # Get the matrice P for the solver
    def __get_quad(self, data, n, l):
        p = (n,n)
        P = numpy.zeros(p)
        for i in xrange(0, n):
            for j in xrange(i, n):
                # (see Annexe page 3 formula 8)
                val = l[i] * l[j] * self.kernel(data[i], data[j])
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
        A = matrix(labels, (1,n))
        r = qp(matrix(P), matrix(q), matrix(G), matrix(h), A, matrix(b))
        # Get alpha vector
        al = numpy.array(r['x'])
        al = al.round()
        # Delete 0 value
        alpha = [(int(al[i]), i)  for i in xrange(0, len(al)) if al[i] <> 0]
        return alpha

    def print_2Ddecision(self, data, lab, bounds, print_sv=True, print_non_sv=False):
        i = 0
        pylab.figure()

        for i in xrange(0, len(lab)):
            if lab[i] >= 0:
                pylab.scatter(data[i][0], data[i][1], c="r", marker='o')
            else:
                pylab.scatter(data[i][0], data[i][1], c="b", marker='o')

        X = []
        Y = []
        for a in self.alpha:
            X.append(data[a[1]][0])
            Y.append(data[a[1]][1])
            pylab.scatter(X, Y, s=200, c="r", marker='o')

        pylab.axis([-max_xy - 1, max_xy + 1, -max_xy - 1, max_xy + 1])
        pylab.savefig(filename + ".pdf")
        pylab.savefig(filename + ".eps")
        pylab.savefig(filename + ".svg")

        return

def __test():
    print "Testing SVM..."
    data, labels = pickle.load(open(filename + ".bin", "r"))

    l = []
    for i in xrange(0, len(labels)):
        if labels[i] == 'outside':
            l.append(1.)
        else:
            l.append(-1.)
# Tiny probleme
#     d = numpy.array([[1], [2], [3]])
#     l = [-1.0, 1.0, 1.0]

    svm = SVM();
    svm.train(data, l)
    lab = svm.process(data)

    svm.print_2Ddecision(data, lab, 1)
if test:
    __test()
