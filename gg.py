__author__ = 'tweninge' + '@' + 'nd.edu'
__author__ = 'saguinag' + '@' + 'nd.edu'
__author__ = 'rodrigopala91' + '@' + 'gmail.com'
__version__ = "0.1.0"

##
##  hrgm_ns = hyperedge replacement grammars model
##  http://connor-johnson.com/2014/02/18/linear-regression-with-python/
##  http://iacs-courses.seas.harvard.edu/courses/am207/blog/lecture-5.html


# TODO: Sort out the returns
# VersionLog:
# 0.0.1 Initial state; modified tw_karate_chop to accommodate this version
#

import os
import re
import math
import networkx as nx
import numpy as np
import tw_karate_chop as tw
import net_metrics as metrics
import graph_sampler as gs
import david as pcfg
import product
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

prod_rules = {}
debug = False


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


def matcher(lhs, N):
    if lhs == "S":
        return [("S", "S")]
    m = []
    for x in N:
        if len(x) == lhs.count(",") + 1:
            i = 0
            for y in lhs.split(","):
                m.append((x[i], y))
                i += 1
            return m


def binarize(tree):
    (node, children) = tree
    children = [binarize(child) for child in children]
    if len(children) <= 2:
        return (node, children)
    else:
        # Just make copies of node.
        # This is the simplest way to do it, but it might be better to trim unnecessary members from each copy.
        # The order that the children is visited is arbitrary.
        binarized = (node, children[:2])
        for child in children[2:]:
            binarized = (node, [binarized, child])
        return binarized



def grow(rule_list, grammar, diam=0):
    D = list()
    newD = diam
    H = list()
    N = list()
    N.append(["S"])  # starting node
    ttt = 0
    # pick non terminal
    num = 0
    for r in rule_list:
        rule = grammar.by_id[r][0]
        lhs_match = matcher(rule.lhs, N)
        e = []
        match = []
        for tup in lhs_match:
            match.append(tup[0])
            e.append(tup[1])
        lhs_str = "(" + ",".join(str(x) for x in sorted(e)) + ")"

        new_idx = {}
        n_rhs = rule.rhs
        # print lhs_str, "->", n_rhs
        for x in n_rhs:
            new_he = []
            he = x.split(":")[0]
            term_symb = x.split(":")[1]
            for y in he.split(","):
                if y.isdigit():  # y is internal node
                    if y not in new_idx:
                        new_idx[y] = num
                        num += 1
                        if diam > 0 and num >= newD and len(H) > 0:
                            newD = newD + diam
                            newG = nx.Graph()
                            for e in H:
                                if (len(e) == 1):
                                    newG.add_node(e[0])
                                else:
                                    newG.add_edge(e[0], e[1])
                                    # D.append(metrics.bfs_eff_diam(newG, 20, 0.9))
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
                #print n_rhs, new_he
        match = sorted(match)
        N.remove(match)

    newG = nx.Graph()
    for e in H:
        if (len(e) == 1):
            newG.add_node(e[0])
        else:
            newG.add_edge(e[0], e[1])

    return newG, D

def graphical_degree_sequence(std_distr, x_mult):
  if std_distr is None: return

  #pois_sequence = np.random.poisson(4.588, x_mult)
  # val_lambda = 2.435294
  # pois_sequence = np.random.poisson(val_lambda, x_mult)
  # while nx.is_valid_degree_sequence_havel_hakimi(pois_sequence):
  #   pois_sequence = np.random.poisson(val_lambda, x_mult)
  val_lambda = 0.291095890
  sequence_ar = np.random.geometric(val_lambda, x_mult)
  while nx.is_valid_degree_sequence_havel_hakimi(sequence_ar):
    sequence_ar = np.random.geometric(val_lambda, x_mult)
  return sequence_ar



# ~~~~~~~~~~~~~
# Main - Begin
'''
import shelve
shelf = shelve.open("../Results/production_rules_dict.shl.db") # the same filename that you used before, please
prod_rules = shelf["karate"]
shelf.close()


rules = []
id = 0
for k, v in prod_rules.iteritems():
    sid = 0
    for x in prod_rules[k]:
        rhs = re.findall("[^()]+", x)
        rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
        #print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])
        sid += 1
    id += 1

g = pcfg.Grammar('S')
for (id, lhs, rhs, prob) in rules:
    g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

print "Starting max size"

G = nx.karate_club_graph()
n = G.number_of_nodes()
degree_sequence = G.degree().values()
#print degree_sequence

clgraph = nx.expected_degree_graph(degree_sequence)

f, axs = plt.subplots(1, 1, figsize=(1.6*6., 1*6.))

target_nodes = 256
g.set_max_size(target_nodes)
print "Done with max size"
# graphletG = []
# graphletH = []
multiGraphs   = []
chungluGraphs = []
kronGraphs    = []

for i in range(100):
    rule_list = g.sample(target_nodes)
    #print rule_list#type(rule_list),'rule_list', len(rule_list)
    # f.write("\n".join(map(lambda x: str(x), rule_list)))
    # PHRG
    hstar = grow(rule_list, g)[0]
    # CLGM
    z = graphical_degree_sequence(target_nodes)
    cl_grph = nx.expected_degree_graph(z)
    # KPGM -
    k = int(math.log(target_nodes,2))
    # from: i:/data/saguinag/Phoenix/demo_graphs/karate.txt 
    P = [[0.9999,0.661],[0.661,     0.01491]]
    KPG = product.kronecker_random_graph(k,P)
    for u,v in KPG.selfloop_edges():
      KPG.remove_edge(u,v)

    multiGraphs.append(hstar)
    chungluGraphs.append(cl_grph)
    kronGraphs.append(KPG)

    # print "H* nodes: " + str(hstar.number_of_nodes())
    # print "H* edges: " + str(hstar.number_of_edges())
    # graphletG.append(gs.subgraphs_cnt(G, 100))
    # graphletH.append(gs.subgraphs_cnt(hstar, 100))
  #
  # metrics.draw_graphlet_plot(graphletG, graphletH)
metrics.draw_degree_probability_distribution(G, multiGraphs,   axs, 'b','b',gname='PHRG')
metrics.draw_degree_probability_distribution(G, chungluGraphs, axs, 'r','r',gname='ChungLu')
metrics.draw_degree_probability_distribution(G, kronGraphs,    axs, 'g','g', gname='KronProd')


plt.legend(labels=['Orig Graph','PHRG','Chung-Lu','KronProd'])

plt.savefig('outfig', bb_inches='tight')
plt.close()
print 'Done'
'''
