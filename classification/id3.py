# id3.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import copy
import sys
import pickle

if len(sys.argv) >= 2:
    filename = sys.argv[1]

test = 0
if sys.argv[len(sys.argv) - 1] == '--test':
    test = 1

def entropy(labels):
    ld = {}
    result = 0.
    sz = len(labels)
    # Only take into account the classes that are present
    # (pi = 0 => -pi log2 pi = 0)
    for l in labels:
        if l in ld:
            ld[l] += 1.
        else:
            ld[l] = 1.
    for v in ld.iteritems():
        pi = v[1] / sz
        result += - (pi * math.log(pi, 2))
    return result

class Tree:
    def __init__(self):
        return

#
# ID3 class definition
#
class ID3(object):
    def __init__(self, verbose = 0):
        self.verbose = verbose
        self.tree = Tree()
        self.tree.etp = 0.
        self.tree.gain = -1.
        self.tree.n = -1
        self.tree.child = {}
        self.tree.label = ""
        return

    def __split(self, data, labels):
        result = Tree()
        result.etp = entropy(labels) # try not to recompute
        result.gain = -1.
        result.n = -1
        result.child = {}
        result.label = ""
        best_dict = {}
        best_d_dict = {}
        if len(data) == 1 or result.etp == 0.:
            result.label = labels[0]
            return result
        for n in range(len(data[0])):
            dict = {}
            d_dict = {}
            i = 0
            for d in data:
                l = d[n]
                if l in dict:
                    dict[l].append(labels[i])
                    d_dict[l].append(d)
                else:
                    dict[l] = [labels[i]]
                    d_dict[l] = [d]
                i += 1
            etp = 0.
            if len(dict.keys()) <= 1:
                continue
            for lbls in dict.iteritems():
                etp += entropy(lbls[1]) * len(lbls[1]) / len(data)
            gain = result.etp - etp
            if gain > result.gain:
                best_dict = dict
                best_d_dict = d_dict
                result.gain = gain
                result.n = n
        if result.n == -1:
            if self.verbose:
                print "ID3.train: Training error (inconsistent data)."
                print data
                print labels
            result.label = labels[len(labels) - 1] # Inconsistent data?
            return result
        for l in best_dict.keys():
            result.child[l] = self.__split(best_d_dict[l], best_dict[l])
        return result

    def __tr(self, i, n, tr):
        d = []
        d.append({-1:"Outlook", 0:"Overcast", 1:"Rain", 2:"Sunny"})
        d.append({-1:"Temperature", 0:"Cool", 1:"Hot", 2:"Mild"})
        d.append({-1:"Humidity", 0:"High", 1:"Normal"})
        d.append({-1:"Wind", 0:"Strong", 1:"Weak"})
        if tr and len(d) > n:
            if i in d[n]:
                return d[n][i]
        if i == -1:
            return str(n)
        return str(i)

    def __print(self, tree, prefix, tr):
        if tree.n == -1:
            print prefix, tree.label
            return
        print prefix, "[Entropy:", tree.etp, "] [Gain: ", tree.gain, "]"
        for l in tree.child.keys():
            self.__print(tree.child[l],
                         prefix + " (" + self.__tr(-1, tree.n, tr) + ") "
                         + self.__tr(l, tree.n, tr) + " >",
                         tr)

    def __print_latex(self, tree, prefix, suffix, tr):
        if tree.n == -1:
            print prefix + "\\TR{" + tree.label + " (entropy: " \
                + str(tree.etp) + ")}\\thput{" + suffix + "}"
            return
        print prefix + "\\pstree{\\TR{" \
            + self.__tr(-1, tree.n, tr) \
            + "? (entropy: " + str(tree.etp) + ", gain: " \
            + str(tree.gain) + ")}\\thput{" + suffix + "}}"
        print prefix + "{"
        for l in tree.child.keys():
            self.__print_latex(tree.child[l],
                               prefix + "  ",
                               self.__tr(l, tree.n, tr),
                               tr)
        print prefix + "}"

    def train(self, data, labels):
        self.tree = self.__split(data, labels)

    def print_tree(self, translate_names):
        self.__print(self.tree, "", translate_names)

    def print_tree_latex(self, translate_names):
        print "\\pstree[levelsep=20ex]{\\TR{" \
            + self.__tr(-1, self.tree.n, translate_names) \
            + "? (entropy: " + str(self.tree.etp) + ", gain: " \
            + str(self.tree.gain) + ")}}"
        print "{"
        for l in self.tree.child.keys():
            self.__print_latex(self.tree.child[l],
                               "  ",
                               self.__tr(l, self.tree.n, translate_names),
                               translate_names)
        print "}"

    def __find(self, tree, v):
        if tree.n == -1:
            return tree.label
        else:
            if v[tree.n] in tree.child:
                return self.__find(tree.child[v[tree.n]], v)
        return ""

    def process(self, data):
        result = []
        i = 0
        for v in data:
            result.append((self.__find(self.tree, v), 1))
            i += 1

        return result

def __test():
    print "Testing ID3..."
    data, labels = pickle.load(open(filename + ".bin", "r"))
    cl = ID3(1)
    cl.train(data, labels)
    print "Built tree:"
    tr = 0
    if (filename == "tennis_data"):
        tr = 1
    cl.print_tree(tr)
    print "Latex source code:"
    print "--------------------------------------------------------------"
    cl.print_tree_latex(tr)
    print "--------------------------------------------------------------"
    print "Checking consistency..."
    lb = cl.process(data)
    consistent = 1
    for i in range(len(lb)):
        if labels[i] != lb[i][0]:
            consistent = 0
    if consistent:
        print "* Passed"
    else:
        print "* Failed"
        print "  Original labels:"
        print labels
        print "  Classification results:"
        print lb

if test:
    __test()
