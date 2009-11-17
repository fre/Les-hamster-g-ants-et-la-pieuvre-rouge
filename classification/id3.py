# id3.py - TP MLEA d-hall_f
#
# See comments within the source code.

import numpy
import math
import copy
import sys
import pickle
import discretize
import os

if len(sys.argv) >= 2:
    filename = sys.argv[1]

def gain(dict, data, entro):
    etp = 0.
    for lbls in dict.iteritems():
        norm = float(len(lbls[1])) / float(len(data))
        etp += entropy(lbls[1])[0] * norm
    gain = entro - etp
    return gain

def gainratio(dict, data, entro):
    etp = 0.
    splitinfo = 0.
    for lbls in dict.iteritems():
        norm = float(len(lbls[1])) / float(len(data))
        etp += entropy(lbls[1])[0] * norm
        splitinfo -= (math.log(norm, 2)) * norm
    gain = entro - etp
    gainratio = gain / splitinfo
    return gainratio

def gini(dict, data, entro):
    etp = 1.
    for lbls in dict.iteritems():
        norm = float(len(lbls[1])) / float(len(data))
        etp -= norm**2
    return etp

test = 0
prune = 0
func = gain
discrf = discretize.Discretize.default
arg = -1
while -arg < len(sys.argv):
    if sys.argv[arg] == '--test':
        test = 1
    if sys.argv[arg] == '--gain':
        func = gain
    if sys.argv[arg] == '--gainratio':
        func = gainratio
    if sys.argv[arg] == '--gini':
        func = gini
    if sys.argv[arg] == '--ewd':
        discrf = discretize.Discretize.ewd
    if sys.argv[arg] == '--efd':
        discrf = discretize.Discretize.efd
    if sys.argv[arg] == '--prune':
        prune = 1
    arg -= 1

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
    return result, ld

class Tree:
    def __init__(self):
        return


#
# ID3 class definition
#
class ID3(object):
    def __init__(self, verbose = 0, fentropy=gain, prune=0, attr_names=[]):
        self.verbose = verbose
        self.fentropy = fentropy
        self.tree = Tree()
        self.tree.etp = 0.
        self.tree.gain = -1.
        self.tree.n = -1
        self.tree.child = {}
        self.tree.label_counts = {}
        self.tree.label = ""
        self.tree.plabel = 1.
        self.attr_names = attr_names
        self.prune = prune
        return

    def tree_size(self, tree):
        size = 0
        for v, c in tree.child.iteritems():
            size += self.tree_size(c)
        return size + 1

    def __split(self, data, labels):
        result = Tree()
        result.etp, result.label_counts = entropy(labels)
        result.gain = -1.
        result.n = -1
        result.child = {}
        best_dict = {}
        best_d_dict = {}
        result.label = labels[0]
        result.plabel = 1.

        if len(data) == 1 or result.etp == 0.:
            return result

        # Compute default label (the most present)
        max = 0
        for k, p in result.label_counts.iteritems():
            if p > max:
                result.label = k
                result.plabel = p / float(len(labels))
                max = p

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
            splitinfo = 0.
            gimpurity = 1.
            if len(dict.keys()) <= 1:
                continue

            gain = self.fentropy(dict, data, result.etp)
            if gain > result.gain:
                best_dict = dict
                best_d_dict = d_dict
                result.gain = gain
                result.n = n

        if result.n == -1:
            # Inconsistent data
            return result
        for l in best_dict.keys():
            result.child[l] = self.__split(best_d_dict[l], best_dict[l])
        return result

    def __tr(self, i, n, tr):
        d = self.attr_names
        if tr and len(d) > n:
            if i in d[n]:
                return d[n][i]
        if i == -1:
            return str(n)
        return str(i)

    # REP pruning
    def __prune(self, tree, data, labels):
        def eval_error(tree):
            error = 0.
            i = 0
            for v in data:
                lc, lp = self.__find(tree, v)
                if lc != labels[i]:
                    error += 1.
                i += 1
            return error

        # Evaluate the error on all tree obtained by pruning one node
        # the root tree.
        def recurse(node, candidate_tree, candidate_error):
            if node.n == -1:
                return
            else:
                child = node.child
                n = node.n
                node.n = -1
                node.child = {}
                error = eval_error(tree)
                if error <= candidate_error[0]:
                    candidate_error[0] = error
                    candidate_tree[0] = copy.deepcopy(tree)
                node.n = n
                node.child = child
                for v, c in node.child.iteritems():
                    recurse(c, candidate_tree, candidate_error)

        error = eval_error(tree)
        candidate_tree = [tree]
        candidate_error = [error]
        iter = 0
        while (candidate_error[0] <= error):
            error = candidate_error[0]
#             print "Iteration " + str(iter)
#             print "Error: " + str(candidate_error[0])
#             print "Candidate: " + str(candidate_tree[0])
            recurse(tree, candidate_tree, candidate_error)
            if (tree != candidate_tree[0]):
                tree = candidate_tree[0]
            else:
                break
            iter += 1

        return candidate_tree[0]

    def train(self, data, labels):
        if prune == 0 or len(labels) < 12:
            self.tree = self.__split(data, labels)
        else:
            # 25% of the labels go for pruning.
            size = len(labels)
            prune_index = len(data) * 1 / 4
            tmax = self.__split(data[0:prune_index], labels[0:prune_index])
            self.tree = self.__prune(tmax, data[prune_index + 1: size - 1],
                                     labels[prune_index + 1: size - 1])

    def __print(self, fd, tree, prefix, tr):
        if tree.n == -1:
            print >>fd, prefix, tree.label
            return
        print >>fd, prefix, "[Entropy:", tree.etp, "] [Gain: ", tree.gain, "]"
        for l in tree.child.keys():
            self.__print(fd,
                         tree.child[l],
                         prefix + " (" + self.__tr(-1, tree.n, tr) + ") "
                         + self.__tr(l, tree.n, tr) + " >",
                         tr)

    def __print_latex(self, fd, tree, prefix, suffix, tr):
        if tree.n == -1:
            print >>fd, prefix + "\\TR{" + tree.label + " (entropy: " \
                + str(tree.etp) + ")}\\thput{" + suffix + "}"
            return
        print >>fd, prefix + "\\pstree{\\TR{" \
            + self.__tr(-1, tree.n, tr) \
            + "? (entropy: " + str(tree.etp) + ", gain: " \
            + str(tree.gain) + ")}\\thput{" + suffix + "}}"
        print >>fd, prefix + "{"
        for l in tree.child.keys():
            self.__print_latex(fd,
                               tree.child[l],
                               prefix + "  ",
                               self.__tr(l, tree.n, tr),
                               tr)
        print >>fd, prefix + "}"

    def __print_dot(self, fd, tree, tr, id):
        id[0] += 1
        num = id[0]
        if tree.n == -1:
            print >>fd, str(num) + "[shape=ellipse, label=\"" + tree.label \
                + "\\nEtp: " \
                + ("%4.3f" % tree.etp) + "\"];"
            return
        print >>fd, str(num) + "[label=\"" \
            + self.__tr(-1, tree.n, tr) \
            + "?\\nEtp: " + ("%4.3f" % tree.etp) + "\\nGain: " \
            + ("%4.3f" % tree.gain) + "\"];"
        for l in tree.child.keys():
            numc = id[0] + 1
            self.__print_dot(fd, tree.child[l], tr, id)
            print >>fd, str(num) + "->" + str(numc) + "[label=\"" \
                + self.__tr(l, tree.n, tr) + "\"];"

    def print_tree(self, fd, translate_names):
        self.__print(fd, self.tree, "", translate_names)

    def print_tree_dot(self, fd, translate_names):
        print >>fd, "digraph G {"
        print >>fd, "label=\"Decision tree for " + filename \
            + "\\nTree size = " + str(self.tree_size(self.tree)) + "\";"
        print >>fd, "node[shape=box];"
        self.__print_dot(fd, self.tree, translate_names, [0])
        print >>fd, "}"

    def print_tree_latex(self, fd, translate_names):
        print >>fd, "\\pstree[levelsep=20ex]{\\TR{" \
            + self.__tr(-1, self.tree.n, translate_names) \
            + "? (entropy: " + str(self.tree.etp) + ", gain: " \
            + str(self.tree.gain) + ")}}"
        print >>fd, "{"
        for l in self.tree.child.keys():
            self.__print_latex(fd,
                               self.tree.child[l],
                               "  ",
                               self.__tr(l, self.tree.n, translate_names),
                               translate_names)
        print >>fd, "}"

    def __find(self, tree, v):
        if tree.n == -1:
            return (tree.label, tree.plabel)
        else:
            if v[tree.n] in tree.child:
                return self.__find(tree.child[v[tree.n]], v)
            else:
                return (tree.label, tree.plabel)

    def process(self, data):
        result = []
        i = 0
        for v in data:
            result.append(self.__find(self.tree, v))
            i += 1

        return result

def load_names(filename):
    res = []
    if not os.path.exists(filename):
        return res
    fh = open(filename, "r")
    fl = fh.readlines()
    for line in fl:
        res.append({-1:line.split('\n')[0]})
    return res

def __test():
    print "Testing ID3..."
    data, labels = pickle.load(open(filename + ".bin", "r"))

    names = []
    if filename == "tennis_data":
        names.append({-1:"Outlook", 0:"Overcast", 1:"Rain", 2:"Sunny"})
        names.append({-1:"Temperature", 0:"Cool", 1:"Hot", 2:"Mild"})
        names.append({-1:"Humidity", 0:"High", 1:"Normal"})
        names.append({-1:"Wind", 0:"Strong", 1:"Weak"})
    else:
        names = load_names(filename + ".names")

    data = discrf(discretize.Discretize(), data)

    cl = ID3(1, func, prune, names)
    cl.train(data, labels)
    print "Built tree:"
    tr = 0
    if (names != []):
        tr = 1
    fdout = open("/dev/stdout", "w")
    fddot = open(filename + "_graph.dot", "w")
    cl.print_tree(fdout, tr)
    print "Latex tree code:"
    print "--------------------------------------------------------------"
    cl.print_tree_latex(fdout, tr)
    print "Dot tree code:"
    print "--------------------------------------------------------------"
    cl.print_tree_dot(fdout, tr)
    cl.print_tree_dot(fddot, tr)
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

    print cl.tree_size(cl.tree)

if test:
    __test()
