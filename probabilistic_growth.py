__author__ = 'tweninge'

import itertools
import random
import collections
import numpy as np


def partitions(n, k):
    """Generate k-tuples of positive integers that sum to n."""
    assert k >= 1
    assert n >= 1
    for comb in itertools.combinations(range(1, n), k - 1):
        part = []
        prev = 0
        for cur in comb:
            part.append(cur - prev)
            prev = cur
        part.append(n - prev)
        yield tuple(part)


def constrain(g, n):
    """Constrains g to generate only strings of length n.
    Warning: This is exponential in the number of nonterminals
    in a rhs of g, which is nonoptimal if >2."""

    nonterminals = {lhs for _, lhs, _, _ in g}
    nonterminals_size = set()

    gc = collections.defaultdict(list)

    for r, lhs, rhs, p in g:
        num_terminals = len([x for x in rhs if x not in nonterminals])
        num_nonterminals = len(rhs) - num_terminals
        if num_nonterminals > 0:
            for size in xrange(num_terminals + 1, n + 1):
                lhs_size = "{}_{}".format(lhs, size) if size < n else lhs
                for nt_sizes in partitions(size - num_terminals, num_nonterminals):
                    rhs_size = []
                    i = 0
                    for x in rhs:
                        if x in nonterminals:
                            x_size = "{}_{}".format(x, nt_sizes[i])
                            rhs_size.append(x_size)
                            nonterminals_size.add(x_size)
                            i += 1
                        else:
                            rhs_size.append(x)
                    gc[lhs_size].append((r, rhs_size, p))
        else:
            lhs_size = "{}_{}".format(lhs, num_terminals) if num_terminals < n else lhs
            gc[lhs_size].append((r, rhs, p))

    # Since our representation of CFG uses the left-hand sides to decide
    # what's a nonterminal, add dummy rules for all nonterminals

    for lhs in nonterminals_size:
        if lhs not in gc:
            gc[lhs].append((None, ["dummy"], 0.0))

    return [(r, lhs, rhs, p) for lhs in gc for (r, rhs, p) in gc[lhs]]


class AliasMethod(object):
    """Sample from a categorical distribution in constant time.

    From https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/"""

    def __init__(self, probs):
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)

        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger = []
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)

        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()

            J[small] = large
            q[large] = q[large] - (1.0 - q[small])

            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)

        self.J = J
        self.q = q

    def sample(self):
        J = self.J
        q = self.q

        K = len(J)

        # Draw from the overall uniform mixture.
        kk = int(np.floor(np.random.rand() * K))

        # Draw from the binary mixture, either keeping the
        # small one, or choosing the associated larger one.
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]


class Sampler(object):
    def __init__(self, g):
        """g is a probabilistic CFG."""
        self.g = collections.defaultdict(list)
        self.p = collections.defaultdict(list)

        # Index rules and their probabilities
        for r, lhs, rhs, p in g:
            self.g[lhs].append((r, rhs))
            self.p[lhs].append(p)
        for lhs in self.p:
            self.p[lhs] = AliasMethod(self.p[lhs])

    def sample(self, start):
        if start not in self.g:
            print "Nope"
            return [], []
            # raise ValueError("can't generate a {}".format(start))
        w = [start]
        i = 0
        rules = []
        while i < len(w):
            if w[i] in self.g:
                lhs = w[i]
                k = self.p[lhs].sample()
                rule, rhs = self.g[lhs][k]
                rules.append(rule)
                w[i:i + 1] = rhs
            else:
                i += 1
        return rules, w


def renormalize(g):
    """Convert a weighted CFG into a probabilistic CFG."""
    # Index the rules
    index = collections.defaultdict(list)
    for r, lhs, rhs, p in g:
        index[lhs].append((r, rhs, p))
    dep = {x: set() for x in index}
    for r, lhs, rhs, p in g:
        for x in rhs:
            if x in index:
                dep[lhs].add(x)

    # Topologically sort the nonterminals
    topological = []
    while len(dep) > 0:
        for x in dep:
            if len(dep[x]) == 0:
                topological.append(x)
                for y in dep:
                    dep[y].discard(x)
                break
        else:
            raise ValueError("cycle detected")
        del dep[x]

    # Compute inside weights
    inside = collections.defaultdict(float)
    for x in topological:
        for r, rhs, p in index[x]:
            for y in rhs:
                if y in index:
                    p *= inside[y]
            inside[x] += p

    # Adjust rule probabilities
    gn = []
    for lhs in index:
        if inside[lhs] > 0.:
            for r, rhs, p in index[lhs]:
                for y in rhs:
                    if y in index:
                        p *= inside[y]
                p /= inside[lhs]
                if p > 0.:
                    gn.append((r, lhs, rhs, p))

    return gn


def constrained_sampler(g, n):
    gc = constrain(g, n)
    gcn = renormalize(gc)
    return Sampler(gcn)


def try_combination(lhs, N):
    lhs
    for i in range(0, len(N)):
        n = N[i]
        if lhs[0] == "S":
            break
        if len(lhs) == len(n):
            t = []
            for i in n:
                t.append(i)
            random.shuffle(t)
            return zip(t, lhs)
    return []

n_distribution = {}
def grow(rule_probabilities, prod_rule_set, n):
    print "sampling with length", n
    s = constrained_sampler(rule_probabilities, n)
    for i in xrange(1):
        rules, words = s.sample("S")
        # print " ".join(words)

        rule_idx = {}
        for r in g:
            rule_idx[r[0]] = r[1:3]

        N = list()
        H = list()
        num = 0
        for r in rules:
            applied_rule = rule_idx[r][0]
            lhs, rhs = prod_rule_set[r]

            lhs = lhs.lstrip("(").rstrip(")")
            lhs_ar = lhs.split(",")
            lhs_match = try_combination(lhs_ar)

            match = []
            for tup in lhs_match:
                match.append(tup[0])

            new_idx = {}
            rhs = rhs.lstrip("(").rstrip(")")
            # print lhs, "->", rhs
            for x in rhs.split(")("):
                new_he = []
                he = x.split(":")[0]
                term_symb = x.split(":")[1]
                for y in he.split(","):
                    if y.isdigit():  # y is internal node
                        if y not in new_idx:
                            new_idx[y] = num
                            num += 1
                        new_he.append(new_idx[y])
                    else:  # y is external node
                        for tup in lhs_match:  # which external node?
                            if tup[1] == y:
                                new_he.append(tup[0])
                                break
                # prod = "(" + ",".join(str(x) for x in new_he) + ")"
                if term_symb == "N":
                    N.append(sorted(new_he))
                elif term_symb == "T":
                    H.append(new_he)

            if len(match) > 0: N.remove(sorted(match))

        newG = nx.Graph()
        for e in H:
            if (len(e) == 1):
                newG.add_node(e[0])
            else:
                newG.add_edge(e[0], e[1])

                # n = newG.number_of_nodes()
                # if n in n_distribution:
                #    n_distribution[newG.number_of_nodes()] += 1
                # else:
                #    n_distribution[newG.number_of_nodes()] = 1

    for k in sorted(n_distribution.keys()):
        print k, "\t", n_distribution[k]