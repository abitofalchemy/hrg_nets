import networkx as nx
import os
import tree_decomposition as td
import PHRG as hrg

def graph_checks(G):
    ## Target number of nodes
    global num_nodes
    num_nodes = G.number_of_nodes()

    if not nx.is_connected(G):
        print "Graph must be connected";
        os._exit(1)

    if G.number_of_selfloops() > 0:
        print "Graph must be not contain self-loops";
        os._exit(1)


G = nx.karate_club_graph()

prod_rules = {}

if 0:
    print '\t',
    print G.number_of_nodes()
    print '\t',
    print G.number_of_edges()

G.remove_edges_from(G.selfloop_edges())
giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
G = nx.subgraph(G, giant_nodes)
num_nodes = G.number_of_nodes()

graph_checks(G)

print
print "--------------------"
print "-Tree Decomposition-"
print "--------------------"

if num_nodes >= 500:
    for Gprime in gs.rwr_sample(G, 2, 100):
        T = td.quickbb(Gprime)
        root = list(T)[0]
        T = td.make_rooted(T, root)
        T = hrg.binarize(T)
        root = list(T)[0]
        root, children = T
        td.new_visit(T, G, prod_rules)
else:
    T = td.quickbb(G)
    root = list(T)[0]
    T = td.make_rooted(T, root)
    T = hrg.binarize(T)
    root = list(T)[0]
    root, children = T
    td.new_visit(T, G, prod_rules)

def flatten(tup):
    if type(tup) == frozenset:
        print type(tup)
    else:
        print type(tup[0]), type([1])
        return flatten(tup[0]), flatten(tup[1])

def listit(t):
    return list(map(listit, t)) if isinstance(t, (list, tuple)) else len(t)


import numpy as np
import tst
# # print tst.s0flat(T)
print '... Treewidth:', np.max([len(x) for x in tst.flatten(T)])-1
