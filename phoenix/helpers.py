from __future__ import division
from .SortedKeyCollections import SortedKeyDefaultDict, SortedKeyCounter

def intersection_size(clique_one, clique_two):
    #print(set(clique_one).intersection(clique_two))
    return len(set(clique_one).intersection(clique_two))

def max_intersecting_clique(clq, clqs):
    """
    return the clique (from *clqs*) that produces
    the largest intersection with *clq*
    """
    clqs_max_intx_dict = dict()
    j = 0
    for c in clqs:
        sz_intx = len(set(c).intersection(clq))
        clqs_max_intx_dict[sz_intx] = j
        j += 1
    if not max(clqs_max_intx_dict.keys()):
        #print '\t No intersection',max(clqs_max_intx_dict.keys())
        max_intxn = []
    else:
        clqs_i =  clqs_max_intx_dict[max(clqs_max_intx_dict.keys())]
        #print '\t',clq, clqs[clqs_i]

    #max_intxn = max(clqs, key=lambda x: intersection_size(x,clq))
    #print max(clqs_max_intx_dict.keys()),'<- max clique size\n',clq,'<- clq\n', clqs[clqs_i],'<- from clqs\n', max_intxn, '<- max_intxn'
    ## Sal: I changed the way return the clique that produces the largest intersection with clq
        max_intxn = clqs[clqs_i]

    return max_intxn

def merge(clq_a, clq_b):
    return set(clq_a).union(clq_b)

def get_weighted_random_value(model):
    """
    This will generate a value given a probability distribution
	(via a continuous distribution function)
    ---
    [P[0][1]] filters the SortedKeyCounter obj
    """
    from bisect import bisect
    from random import random
    #http://stackoverflow.com/questions/4437250/choose-list-variable-given-probability-of-each-variable
    P = list(model.items())
    cdf = [P[0][1]]
    for i in range(1, len(P)):
        cdf.append(cdf[-1] + P[i][1])
    tmp = P[bisect(cdf, random())]
    return P[bisect(cdf, random())][0]

def intersection_size(clique_one, clique_two):
    return len(set(clique_one).intersection(clique_two))

# def max_intersecting_clique(clq, clqs):
#     """
#     return the clique (from *clqs*) that produces
#     the largest intersection with *clq*
#     """
#     max_intxn = max(clqs, key=lambda x: intersection_size(x,clq))
#     return max_intxn

def max_intersection_clique(clq, clqs):
    """
    return the clique (from *clqs*) that produces
    the largest intersection with *clq*
    """
    max_intxn = None
    intxn = None
    max_intxn_size = 0

    if len(clqs) == 1:
        return clqs[0], {}

    for clique in clqs:
        #print("max_intersection_clique:", intxn)
        #intxn_size = len(set(clq).intersection(clique))
        intxn_size = intersection_size(clq, clique)

        if intxn_size > max_intxn_size:
            intxn = set(clq).intersection(clique)
            max_intxn = clique
            max_intxn_size = intxn_size

    return max_intxn, intxn


def intersections_with_hypergraph(central_clique, cliques):
    intxn_counter = collections.defaultdict(collections.Counter)

    for clique in cliques:
        intxn_counter[(len(central_clique), len(clique))][intersection_size(central_clique, clique)] += 1

    return intxn_counter

def merge(clq_a, clq_b):
    return set(clq_a).union(clq_b)


def make_clique_edges(nodes):
    from itertools import combinations
    return list(combinations(nodes, 2))


def clique_number_distribution(cliques):
    clique_number_counts = SortedKeyCounter()

    for clq in cliques:
        clique_number_counts[len(clq)] += 1

    return clique_number_counts

def cliques_x_cliques_distribution(cliques, debug=False):
    """
    Given a list of cliques we enumerate the diff. combinations of 2 unique cliques from the list
    intersection size <- the size of the intersection for the given pair of cliques

    Output: SortedKeyDefaultDict object that counts intersection sizes

    Example: if for the first enum pair their length is 4, having a intersection of 3 similar elements
    the first resulting SKDD would look like: {(4, 4): SortedKeyCounter({3: 1})})
    """
    
    intxn_counters = SortedKeyDefaultDict(SortedKeyCounter)

    from itertools import combinations
    for enum, (clqA, clqB) in enumerate(combinations(cliques, 2)):
        intxn_size = intersection_size(clqA, clqB)

        if debug: print(enum, clqA, clqB, intxn_size)
        # conditional distributions(?)
        if debug: print(intxn_counters)

        #print (enum, intxn_counters[(len(clqA),len(clqB))][intxn_size])
        intxn_counters[(len(clqA),len(clqB))][intxn_size] += 1
        #print ('intersection counters:', intxn_counters)
        #if bool(set(clqA).intersection(clqB)):
        #    print 'Yes!!!'
        #intxn_counters[len(clqA),len(clqB)]['intersects'][intxn_size>0] += 1

    
    return intxn_counters

def normalize_distribution(counter):
    """
    Given a *counter* (frequency distribution,
    AKA collections.Counter), normalize the frequencies
    to produce a probability distribution.
    """
    total = sum(counter.values())
    return counter.__class__({k:v/total for k, v in counter.items()})

def normalize_distributions(distributions):
    return {key:normalize_distribution(val)
            for key, val in distributions.items()}

def normalize_distribution_of_distributions(dist_of_dists):
    return {key: normalize_distributions(val)
            for key, val in dist_of_dists.items()}

