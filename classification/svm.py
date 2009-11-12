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
    return numpy.dot(numpy.transpose(x), y)


def polynomial_k(x, y):
    a = (numpy.dot(numpy.transpose(x), y)) ** int(p_value)
    return a

test = 0
p_value = 1
if sys.argv[-1] == '--test':
    test = 1
if sys.argv[-3] == '--poly_k':
    kernel = polynomial_k
    p_value = sys.argv[-2]
if sys.argv[-2] == '--linear_k':
    kernel = linear_k


class SVM(object):
    def __init__(self, kernel=linear_k):
        self.kernel = kernel
        self.alpha = []
        self.bias = 0.
        self.weight = 0.


    def train(self, data, labels):
        self.alpha = self.lagrange_coeffs(data, labels)
        self.bias = self.bias_get(data, labels)
        print self.bias

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
            deci.append(numpy.dot(numpy.transpose(self.weight), d) + self.bias)
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
        self.weight = self.__weight(data, labels)

        b = 0.
        for i in xrange(0, len(self.alpha)):
            b += (labels[self.alpha[i][1]] - numpy.dot(numpy.transpose(self.weight), data[self.alpha[i][1]]))

        return b / len(self.alpha)


    # Get the matrice P for the solver
    def __get_quad(self, data, n, l):
        p = (n,n)
        P = numpy.zeros(p)
        for i in xrange(0, n):
            for j in xrange(0, n):
                # (see Annexe page 3 formula 8)
                P[i, j] = l[i] * l[j] * self.kernel(data[i], data[j])
        return P

    # get lagrange coefficient (alpha vector)
    # use annexe to understand
    def lagrange_coeffs(self, data, labels):
        n = len(labels)
        q = numpy.dot (numpy.ones(n), -1)
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

    def print_2Ddecision(self, bounds, print_sv=True, print_non_sv=False):
        return

def display(data, labels):
    i = 0
    pylab.figure()
    for d in data:
        if labels[i] == -1:
            pylab.scatter([d[0]], [d[1]], c="r", marker='^')
        else:
            pylab.scatter([d[0]], [d[1]], c="b", marker='o')
        i = i + 1

    pylab.axis([-max_xy - 1, max_xy + 1, -max_xy - 1, max_xy + 1])
    pylab.savefig(filename + ".pdf")
    pylab.savefig(filename + ".eps")
    pylab.savefig(filename + ".svg")

print "Generating " + filename

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

    svm = SVM(polynomial_k);
    svm.train(data, l)
    lab = svm.process(data)

    display(data, lab)

if test:
    __test()
