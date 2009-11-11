# svm.py - TP MLEA
#
# See comments within the source code.

import numpy
import math
import copy
import sys
import cvxopt

from cvxopt.solvers import qp
from cvxopt.base import matrix

if len(sys.argv) >= 2:
    filename = sys.argv[1]

test = 0
if sys.argv[-1] == '--test':
    test = 1

def linear_k(x, y):
    return numpy.dot(numpy.transpose(x), y)

class SVM(object):
    def __init__(self, kernel=linear_k):
        self.kernel = kernel

    def __get_quad(self, data, n, l):
        p = (n,n)
        P = numpy.zeros(p)
        for i in xrange(0, n):
            for j in xrange(0, n):
                P[i, j] = l[i] * l[j] * self.kernel(data[j], data[i])
        return P

    def train(self, data, labels):
        n = len(labels)
        q = numpy.dot (numpy.ones(n), -1)
        h = numpy.zeros(n)
        G = numpy.dot(numpy.eye(n), -1)
        P = self.__get_quad(data, n, labels)
        print P
        print q
        print G
        print h
        r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
        alpha = list(r['x'])
        print alpha
        return

    def decision(self, data):
        return

    def process(self, data):
        return

    def bias_get(self):
        return

    def lagrange_coeffs(self):
        return

    def print_2Ddecision(self, bounds, print_sv=True, print_non_sv=False):
        return

def __test():
    print "Testing SVM..."
    d = numpy.array([[1], [2], [3]])
    l = [-1, 1, 1]

    svm = SVM();
    svm.train(d, l)

if test:
    __test()
