# tennis.py - TP MLEA d-hall_f - cuche_h
#
# See comments within the source code.

import numpy
import math
import pickle
import sys
import shuffle
from matplotlib import pylab

if len(sys.argv) >= 2:
    filename = sys.argv[1]

test = 0
if sys.argv[-1] == '--test':
    test = 1

class Discretize(object):
    def __set_parameters(self, data):
        self.data = data
        self.nb_class = self.__nb_class()


    # Get the max number of class
    def __nb_class(self):
        nb = 1
        nbc = 0
        size = self.data.shape[0]
        # Huntsberger
        nb += (math.log(size, 10)) * 3.3
        return nb
# Ca fait bugger ca non ?
#         for i in xrange(1, int(nb)):
#             if (size % i == 0):
#                 nbc = i
#         if nbc == 1:
#             return int(nb)
#         return nbc

    # sort a dimension
    def __sort_dim(self, n_dim):
        data_sort = [(self.data[j][n_dim], j) for j in xrange(0, self.data.shape[0])]
        data_sort.sort()
        return data_sort

    # mean of a dimension
    def __mean_dim(self, n_dim):
        return numpy.mean(self.data, axis=0)[n_dim]

    # max of a dimension
    def __max_dim(self, n_dim):
        return numpy.max(self.data, axis=0)[n_dim]

    # min of a dimension
    def __min_dim(self, n_dim):
        return numpy.min(self.data, axis=0)[n_dim]

    # Discretize one dimension with ewd algorithm
    def ewd_n_dim(self, n_dim):
        min_bound = self.__min_dim(n_dim)
        max_bound = self.__max_dim(n_dim)
        sig = (max_bound - min_bound) / self.nb_class
        max_bound = min_bound + sig
        sort_data = self.__sort_dim(n_dim)
        l = 1
        new_data = []
        for i in xrange(0, self.data.shape[0]):
            if (sort_data[i][0] <= max_bound):
                new_data.append((sort_data[i][1], l))
            else:
                l += 1
                max_bound += sig
                new_data.append((sort_data[i][1], l))
        new_data.sort()
        dis = [(d[1]) for d in new_data]
        return dis

    # Discretize all the data with ewd algorithm
    def ewd(self, data):
        self.__set_parameters(data)
        new_data = []
        for i in xrange(0, self.data.shape[1]):
            new_data.append(self.ewd_n_dim(i))
        d = [(new_data[0][i], new_data[1][i]) for i in xrange(0, self.data.shape[0])]
        return numpy.array(d)

    # Discretize one dimension with efd algorithm
    def efd_n_dim(self, n_dim):
        nb_element = self.data.shape[0] / self.nb_class
        sort_data = self.__sort_dim(n_dim)
        l = 1
        new_data = []
        for i in xrange(0, self.data.shape[0]):
            if (i < nb_element * l):
                new_data.append((sort_data[i][1], l))
            else:
                l += 1
                new_data.append((sort_data[i][1], l))
        new_data.sort()
        dis = [(d[1]) for d in new_data]
        return dis

    # Discretize all the data with efd algorithm
    def efd(self, data):
        self.__set_parameters(data)
        new_data = []
        for i in xrange(0, self.data.shape[1]):
            new_data.append(self.ewd_n_dim(i))
        d = [(new_data[0][i], new_data[1][i]) for i in xrange(0, self.data.shape[0])]
        return numpy.array(d)

    def default(self, data):
        return data

    def __split(self, cl):
        d = []
        split = len(cl) / 2
        for i in xrange(0, len(cl)):
            if i < split:
                d.append((cl[0][0]*10+1, cl[i][1]))
            else:
                d.append((cl[0][0]*10+2, cl[i][1]))
        return d

    def __merge(self, class1, class2):
        d = [(class1[0][0]*10, class1[i][1]) for i in xrange(0, len(class1))]
        for i in xrange(0, len(class2)):
            d.append((class1[0][0]*10, class2[i][1]))
        return d

    def id_n_dim(self, fpart, n_dim):
        dat = fpart(n_dim)
        dt = [(dat[i], i) for i in xrange(0, len(dat))]
        dt.sort()
        d1 = dt[0:200]
        d2 = dt[201:400]
        self.__split(d1)
