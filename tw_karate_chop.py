#!/Users/saguinag/ToolSet/anaconda/bin/python
##! /usr/bin/python

import sys, os
import networkx as nx
import pandas as pd
import numpy as np
from collections import deque, defaultdict, Counter
import random
import itertools, math
from pprint import pprint
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from datetime import datetime,tzinfo,timedelta
from phoenix import chartvis, load_graph_cliques


plt.style.use('ggplot')

def save_plot_figure_2disk(dsg1=False,plotname=""):
  """
  TODO: plotname TBD
  """
  F = plt.gcf()
  DPI = F.get_dpi()
  plt.axis('off')
  #plt.tight_layout()
  if dsg1:
  	F.savefig("/home/saguinag/public_html/figures/outfig",dpi = DPI,bbox_inches='tight')
  else:
    if not plotname == "":
  	F.savefig(plotname, dpi = DPI,bbox_inches='tight')
    else:
  	F.savefig("../figures/outfig",dpi = DPI,bbox_inches='tight')


def control_rod(H, nbr_of_nodes, choices):
    num_nodes = nbr_of_nodes
    newchoices = []
    p = len(H) / float(num_nodes)
    total = 0

    for i in range(0, len(choices)):
        n = float(choices[i][0].count('N'))
        t = float(choices[i][0].count('T'))

        # 2*(e^-Fx)-1
        x = p

        form = 2 * math.e ** ((-F) * x) - 1
        wn = n * form
        wt = .01  # t*-wn

        ratio = max(0, wt + wn)

        total += ratio
        newchoices.append((choices[i][0], ratio))

    r = random.uniform(0, total)
    upto = 0
    if total == 0:
        random.shuffle(newchoices)
    for c, w in newchoices:
        if upto + w >= r:
            return c
        upto += w
    assert False, "Shouldn't get here"

class Solution(object):
    pass



def make_clique(graph, nodes):
    for v1 in nodes:
        for v2 in nodes:
            if v1 != v2:
                graph[v1].add(v2)

def is_clique(graph, vs):
    for v1 in vs:
        for v2 in vs:
            if v1 != v2 and v2 not in graph[v1]:
                return False
    return True

def simplicial(graph, v):
    return is_clique(graph, graph[v])

def almost_simplicial(graph, v):
    for u in graph[v]:
        if is_clique(graph, graph[v] - {u}):
            return True
    return False

def eliminate_node(graph, v):
    make_clique(graph, graph[v])
    delete_node(graph, v)

def delete_node(graph, v):
    for u in graph[v]:
        graph[u].remove(v)
    del graph[v]

def copy_graph(graph):
    return {u:set(graph[u]) for u in graph}

def contract_edge(graph, u, v):
    """Contract edge (u,v) by removing u"""
    graph[v] = (graph[v] | graph[u]) - {u, v}
    del graph[u]
    for w in graph:
        if u in graph[w]:
            graph[w] = (graph[w] | {v}) - {u, w}

def make_rooted(graph, u, memo=None):
    """Returns: a tree in the format (label, children) where children is a list of trees"""
    if memo is None: memo = set()
    memo.add(u)
    children = [make_rooted(graph, v, memo) for v in graph[u] if v not in memo]
    return (u, children)

def upper_bound(graph):
    """Min-fill."""
    graph = copy_graph(graph)
    dmax = 0
    order = []
    while len(graph) > 0:
        #d, u = min((len(graph[u]), u) for u in graph) # min-width
        d, u = min((count_fillin(graph, graph[u]), u) for u in graph)
        dmax = max(dmax, len(graph[u]))
        eliminate_node(graph, u)
        order.append(u)
    return dmax, order

def count_fillin(graph, nodes):
    """How many edges would be needed to make v a clique."""
    count = 0
    for v1 in nodes:
        for v2 in nodes:
            if v1 != v2 and v2 not in graph[v1]:
                count += 1
    return count/2

def lower_bound(graph):
    """Minor-min-width"""
    graph = copy_graph(graph)
    dmax = 0
    while len(graph) > 0:
        # pick node of minimum degree
        d, u = min((len(graph[u]), u) for u in graph)
        dmax = max(dmax, d)

        # Gogate and Dechter: minor-min-width
        nb = graph[u] - {u}
        if len(nb) > 0:
            _, v = min((len(graph[v] & nb), v) for v in nb)
            contract_edge(graph, u, v)
        else:
            delete_node(graph, u)
    return dmax

def quickbb(graph):
    """Gogate and Dechter, A complete anytime algorithm for treewidth. UAI
       2004. http://arxiv.org/pdf/1207.4109.pdf"""

    """Given a permutation of the nodes (called an elimination ordering),
       for each node, remove the node and make its neighbors into a clique.
       The maximum degree of the nodes at the time of their elimination is
       the width of the tree decomposition corresponding to that ordering.
       The treewidth of the graph is the minimum over all possible
       permutations.
       """

    best = Solution() # this gets around the lack of nonlocal in Python 2
    best.count = 0

    def bb(graph, order, f, g):
        best.count += 1
        if len(graph) < 2:
            if f < best.ub:
                assert f == g
                best.ub = f
                best.order = list(order) + list(graph)
        else:
            vs = []
            for v in graph:
                # very important pruning rule
                if simplicial(graph, v) or almost_simplicial(graph, v) and len(graph[v]) <= lb:
                    vs = [v]
                    break
                else:
                    vs.append(v)

            for v in vs:
                graph1 = copy_graph(graph)
                eliminate_node(graph1, v)
                order1 = order + [v]
                # treewidth for current order so far
                g1 = max(g, len(graph[v]))
                # lower bound given where we are
                f1 = max(g, lower_bound(graph1))
                if f1 < best.ub:
                    bb(graph1, order1, f1, g1)

    graph = { u : set(graph[u]) for u in graph }

    order = []
    best.ub, best.order = upper_bound(graph)
    lb = lower_bound(graph)
    ## 2 lines below commented out by TW to speed up the process
    #if lb < best.ub:
    #    bb(graph, order, lb, 0)

    # Build the tree decomposition
    tree = defaultdict(set)
    def build(order):
        if len(order) < 2:
            bag = frozenset(order)
            tree[bag] = set()
            return
        v = order[0]
        clique = graph[v]
        eliminate_node(graph, v)
        build(order[1:])
        for tv in tree:
            if clique.issubset(tv):
                break
        bag = frozenset(clique | {v})
        tree[bag].add(tv)
        tree[tv].add(bag)
    build(best.order)
    return tree

def make_rooted(graph, u, memo=None):
    """Returns: a tree in the format (label, children) where children is a list of trees"""
    if memo is None: memo = set()
    memo.add(u)
    children = [make_rooted(graph, v, memo) for v in graph[u] if v not in memo]
    return (u, children)

def new_visit(datree, graph, prod_rules, indent=0, parent=None):
    G=graph
    node, subtrees = datree

    itx = parent & node if parent else set()
    rhs = get_production_rule(G, node, itx)
    s = [list(node & child) for child, _ in subtrees]
    add_to_prod_rules(prod_rules, itx, rhs, s)

    #print " "*indent, " ".join(str(x) for x in node)
    for subtree in subtrees:
        tv, subsubtrees = subtree
        new_visit(subtree, G, prod_rules, indent=indent+2, parent=node)

def get_production_rule(G, child, itx):

    #lhs = nx.Graph()
    #for n in G.subgraph(itx).nodes():
    #    lhs.add_node(n)
    #for e in G.subgraph(itx).edges():
    #    lhs.add_edge(e[0], e[1])

    rhs = nx.Graph()
    for n in G.subgraph(child).nodes():
        rhs.add_node(n)
    for e in G.subgraph(child).edges():
        rhs.add_edge(e[0], e[1])

    #remove links between external nodes (edges in itx)
    for x in itertools.combinations(itx,2):
        if rhs.has_edge(x[0],x[1]):
            rhs.remove_edge(x[0], x[1])

    #return lhs, rhs
    return rhs


def add_to_prod_rules(production_rules, lhs, rhs, s):
    prod_rules = production_rules
    letter='a'
    d = {}

    for x in lhs:
        d[x]= letter
        letter=chr(ord(letter) + 1)

    lhs_s = set()
    for x in lhs:
        lhs_s.add(d[x])
    if len(lhs_s) == 0:
        lhs_s.add("S")

    i=0
    rhs_s = nx.Graph()
    for n in rhs.nodes():
        if n in d:
            n = d[n]
        else:
            d[n] = i
            n=i
            i=i+1
        rhs_s.add_node(n)

    for e in rhs.edges():
        u = d[e[0]]
        v = d[e[1]]
        rhs_s.add_edge(u,v)


    lhs_str = "(" + ",".join(str(x) for x in sorted(lhs_s)) + ")"

    nodes = set()
    rhs_term_dict = []
    for c in sorted(nx.edges(rhs_s)):
        rhs_term_dict.append( (",".join(str(x) for x in sorted(list(c))), "T") )
        nodes.add(c[0])
        nodes.add(c[1])

    for c in s:
        rhs_term_dict.append( (",".join(str(d[x]) for x in sorted(c)), "N") )
        for x in c:
            nodes.add(d[x])

    for singletons in set(nx.nodes(rhs_s)).difference(nodes):
        rhs_term_dict.append( ( singletons, "T" ) )

    rhs_str=""
    for n in rhs_term_dict:
        rhs_str = rhs_str + "("+n[0]+":"+n[1]+")"
        nodes.add(n[0])
    if rhs_str=="":
        rhs_str = "()"

    if lhs_str not in prod_rules:
        rhs_dict = {}
        rhs_dict[rhs_str] = 1
        prod_rules[lhs_str] = rhs_dict
    else:
        rhs_dict = prod_rules[lhs_str]
        if rhs_str in rhs_dict:
            prod_rules[lhs_str][rhs_str] = rhs_dict[rhs_str]+1
        else:
            rhs_dict[rhs_str] = 1
        ##sorting above makes rhs match perfectly if a match exists

    print lhs_str, "->", rhs_str


def visit(tu, indent, memo,production_rules, datree, graph):
    G=graph
    T=datree
    prod_rules = production_rules
    if tu in memo:
        return
    memo.add(tu)
    print " "*indent, " ".join(str(x) for x in tu)
    for tv in T[tu]:
        if tv in memo:
            continue
        itx = set(tu).intersection(tv)
        rhs = get_production_rule(G, tv, itx)
        s = list()
        for c in T[tv]:
            if c in memo:  continue
            s.append( list(set(c).intersection(tv)) )
        add_to_prod_rules(prod_rules, itx, rhs, s)
        visit(tv, indent+2, memo,prod_rules,T, G)

def new_visit(datree, graph, prod_rules, indent=0, parent=None):
    G=graph
    node, subtrees = datree

    itx = parent & node if parent else set()
    rhs = get_production_rule(G, node, itx)
    s = [list(node & child) for child, _ in subtrees]
    add_to_prod_rules(prod_rules, itx, rhs, s)

    #print " "*indent, " ".join(str(x) for x in node)
    for subtree in subtrees:
        tv, subsubtrees = subtree
        new_visit(subtree, G, prod_rules, indent=indent+2, parent=node)

def weighted_choice(choices):
  if len(choices) == 1:
    for c in choices:
        #print c[0],':c from weighted choice'
        return c[0]
  total = sum(w for c, w in choices)
  cwN= [(c,w) for c, w in choices if "N)" in c]
  if len(cwN):
    choices.remove(cwN[0])
    choices.append(([val for val,cnt in cwN][0], total))
  population = [val for val, cnt in choices for i in range(cnt)]
  #print ">",population
  cho = random.choice(population)
  #print " ",cho
  return cho

# def weighted_choice(choices):
#   #print choices
#   total = sum(w for c, w in choices)
#   r = random.uniform(0, total)
#   upto = 0
#   for c, w in choices:
#     if upto + w > r:
#         #print c
#         return c
#     upto += w
#   assert False, "Shouldn't get here"


def try_combination(lhs,N):
    for i in range(0,len(N)):
        n = N[i]
        if lhs[0] == "S":
            break
        if len(lhs) == len(n):
            return random.choice( [zip(x,lhs) for x in itertools.permutations(n,len(lhs))] )
    return []

def find_match(N, production_rules):
    prod_rules = production_rules
    if len(N)==1 and ['S'] in N: return [('S','S')]
    matching = {}
    while True:
        #print prod_rules.keys(),':prod rules keys in find match'
        lhs = random.choice(prod_rules.keys()).lstrip("(").rstrip(")") ##TODO make this weighted choice
        lhs = lhs.split(",")
        #print lhs,":lhs"
        c = try_combination(lhs, N)

        if c: return c


def plot_degree_distribution(wiki):
  degs = {}
  for n in wiki.nodes():
    deg = wiki.degree(n)
    if deg not in degs:
      degs[deg] = 0
    degs[deg] += 1
  items = sorted(degs.items())
  items = sorted(degs.items())

  ax.plot([k for (k,v) in items], [v for (k,
                                        v) in items], '-o',alpha=0.75)




F = .6  # math.e

if __name__ == '__main__':

  debug = False

  print
  print "--------------------"
  print "------ Graphs ------"
  print "--------------------"

  ## Board example - Toy Graph

  G = nx.Graph()
  G.add_edge(1,2)
  G.add_edge(2,3)
  G.add_edge(2,4)
  G.add_edge(3,4)
  G.add_edge(3,5)
  G.add_edge(4,6)
  G.add_edge(5,6)
  G.add_edge(5,7)
  G.add_edge(6,7)
  #print '--- Board Example ---'
  grph_name = "Board Ex" # small-world graph

  #G = nx.karate_club_graph()
  #grph_name  = "Karate Club"
  """

  n=68 # 10 nodes
#   m=68 # 20 edges
#   G=nx.gnm_random_graph(n,m) #binomial_graph which is also somtims called the Erdos-Renyi graph.
#   grph_name = "Erdos-Renyi"
# #
  k=2
  p=0.5
  G = nx.newman_watts_strogatz_graph(n, k, p)
#   #G = nx.watts_strogatz_graph(4, 2, 0)
  grph_name = "NwmnWattsStrogatz" # small-world graph

  ## Balanced Tree Graph
  #G=nx.balanced_tree(2,6)
  #print '--- Balalanced Tree Graph ---'
  #save_graph_todisk(G)

  input_graph = "../demo_graphs/politicalbooks.gml"
  G = nx.read_gml(input_graph)
  grph_name = "Politicalbooks"
  """


  print '--- ',grph_name,' ---'
  print '(V,E):', G.number_of_nodes(), G.number_of_edges()

  ## Target number of nodes
  num_nodes = G.number_of_nodes()

  if not nx.is_connected(G) :
    print "Graph must be connected";
    exit()

  if G.number_of_selfloops() > 0 :
    print "Graph must be not contain self-loops";
    exit()


  print
  print "--------------------"
  print "------- Edges ------"
  print "--------------------"

#  for e in G.edges():
#    print e

#  nx.draw_networkx(G)
#  plt.show()

  print
  print "--------------------"
  print "------ Cliques -----"
  print "--------------------"

#  for e in nx.find_cliques(G):
#    print e

  #G = nx.watts_strogatz_graph(4, 2, 0)
  prod_rules={}


  T = quickbb(G)      # tree decomposition
  #print T
  root = list(T)[0]
  #print root, ':root'

#  pprint(list(T)[0])
#  stop()

  print
  print "---------------------"
  print "- Intersection Tree -"
  print "---------------------"

  #get S
  rhs = get_production_rule(G, root, set())
  s = list()
  for c in T[root]:
    s.append( list(set(c).intersection(root)) )
  add_to_prod_rules(prod_rules, set(), rhs, s)
  visit(root, 0, set(),prod_rules)
  exit()
  print
  print "--------------------"
  print "- Production Rules -"
  print "--------------------"

  for k in prod_rules.iterkeys():
    print k
    for d in prod_rules[k]:
        print '\t -> ', d, prod_rules[k][d]

  n_distribution= {}

  eff_diag_run = pd.DataFrame()
  grphs = []

  print
  print "--------------------"
  print "- Runs             -"
  print "--------------------"

  compute_eff_diameter = False
  heterm_s = [] # Stats
  nbr_of_runs = 100
  for run in range(0,nbr_of_runs):
      H = list()
      N = list()
      heterm_cnt = Counter()

      N.append(["S"]) #starting node
      ttt=0

      eff_dia_gph = []
      graphlet_node_cnt = []

      #pick non terminal
      num = 0
      while len(N) > 0:
          lhs_match = find_match(N)
          e = []
          match = []
          for tup in lhs_match:
              match.append(tup[0])
              e.append(tup[1])
          lhs_str = "(" + ",".join(str(x) for x in sorted(e)) + ")"
          #DO SOMETHING USEFUL WITH THIS MATCH
          new_idx = {}
          n_rhs =str(control_rod(H, prod_rules[lhs_str].items())).lstrip("(").rstrip(")")
          #n_rhs =str(weighted_choice(prod_rules[lhs_str].items())).lstrip("(").rstrip(")")
          #n_rhs = str(random.choice(prod_rules[lhs_str].keys())).lstrip("(").rstrip(")")
          # if n_rhs[-1] == "N":
          if not debug: print lhs_str, "->", n_rhs

          for x in n_rhs.split(")("):
              heterm_cnt[x] += 1
              new_he = []
              he = x.split(":")[0]
              term_symb = x.split(":")[1]

              for y in he.split(","):
                  if y.isdigit(): # y is internal node
                      if y not in new_idx:
                          new_idx[y] = num
                          num+=1
                      new_he.append(new_idx[y])
                  else: #y is external node
                      for tup in lhs_match: #which external node?
                          if tup[1] == y:
                              new_he.append(tup[0])
                              break
              #prod = "(" + ",".join(str(x) for x in new_he) + ")"
              if term_symb == "N":
                  N.append(sorted(new_he))
              elif term_symb == "T":
                  H.append(new_he)
          ## ends for
          ##print 'N:',len(N), 'H:',len(H)

          match = sorted(match)
          N.remove(match)

          if compute_eff_diameter:
          	##
          	## EFFECTIVE DIAMETER as graph grows
          	##
						newG = nx.Graph()
						for e in H:
							if (len(e) == 1):
								newG.add_node(e[0])
							else:
								newG.add_edge(e[0], e[1])
						#eff_dia_gph.append(chartvis.bfs_eff_diam(newG, 50, .90))
						#eff_dia_gph.append((newG.number_of_nodes(), chartvis.bfs_eff_diam(newG, 50, .90)))
						eff_dia_gph.append([newG.number_of_nodes(),
															chartvis.bfs_eff_diam(newG, 50, .90)])
      if compute_eff_diameter:
        ed = map(list, zip(eff_dia_gph))
        eff_diag_run = eff_diag_run.append(pd.DataFrame(eff_dia_gph))
	######

      newG = nx.Graph()
      for e in H:
          if(len(e) == 1):
              newG.add_node(e[0])
          else:
              newG.add_edge(e[0], e[1])

      #print len(newG.edges())

      n = newG.number_of_nodes()
      if n in n_distribution:
          n_distribution[newG.number_of_nodes()] += 1
      else:
          n_distribution[newG.number_of_nodes()] = 1

      #if n == num_nodes:
      grphs.append(newG)

      heterm_s.append(heterm_cnt) ## keep each run's count of HETT

      # print run number
      print '\trun:',run,'in',nbr_of_runs
  #------------------------ for loop ends

  if compute_eff_diameter:
    print eff_diag_run.shape
    print eff_diag_run.head()
    ef =  eff_diag_run.groupby([0])

  # print "V = ", newG.number_of_nodes()
  #      print "E = ", newG.number_of_edges()
  #      giant_nodes = max(nx.connected_component_subgraphs(newG), key=len)
  #      giant = nx.subgraph(newG, giant_nodes)
  #      print "V in giant component = ", giant.number_of_nodes()
  #      print "E in giant compenent = ", giant.number_of_edges()
  #      print "Diameter = ", nx.diameter(nx.subgraph(newG, giant))


  ## Print the distribution
  x =[];  y =[]
  for k in sorted(n_distribution.keys()):
    print k,"\t",n_distribution[k]
    x.append(k)
    y.append(n_distribution[k])

  ## Chartvis
  print '-'*80

  ## --------------------------------------------------
  ## Charting and Visualizing the results of the model
#  import numpy.linalg
#  from igraph import *
#  L = nx.laplacian_spectrum(G)
#  #fig, ax = plt.subplots(1, 1, figsize=(1.6*9,1*9))
#  plt.plot(L)

#fig, ax = plt.subplots(2, 4, figsize=(1.6*9,1*9))
  fig = plt.figure(figsize=(1.6*12,1*12))

  ax0 = fig.add_subplot(241)
  ax1 = fig.add_subplot(242)
  ax2 = fig.add_subplot(243)
  ax3 = fig.add_subplot(244)

  ax4 = fig.add_subplot(245)
  ax5 = fig.add_subplot(246)
  ax6 = fig.add_subplot(247)
  ax7 = fig.add_subplot(248)



  chartvis.draw_degree_rank_plot(G, grphs, grph_name, ax0)
  chartvis.draw_scree_plot(      G, grphs, grph_name, ax1)
  chartvis.draw_network_value(   G, grphs, grph_name, ax2)
  chartvis.draw_hop_plot(        G, grphs, grph_name, ax3)
#  ##
  chartvis.draw_nodecount_degreecount_plots(grphs,grph_name, ax4)
  chartvis.draw_degree_distribution(G, grphs, grph_name, ax5)
  #chartvis.Assortativity_Plot(G, grphs, grph_name, ax6)
  chartvis.draw_clustering_coefficients(G,grphs,  grph_name, ax6)

# #  chartvis.draw_nodecount_avgdegree_plots(grphs,  grph_name, ax5)
# #  chartvis.draw_laplacian_spectrum_conf_bands(G,  grphs, grph_name, ax7)
# #  print ef.head(), type(ef)
#
  if compute_eff_diameter:
    ## Effective Diameter Plot
    ax7.plot(ef.mean().index, ef.mean()[1],'b')
    ax7.set_ylabel('Effective Diamater')
    ax7.set_xlabel('Nodes')

  save_plot_figure_2disk()

  ## Counter And Stats:
  pprint(len(heterm_s))
  df = pd.DataFrame(heterm_s)

  fig = plt.figure(figsize=(1.6*5,1*5))
  ax0 = fig.add_subplot(111) #fig, ax = plt.subplots()
  labels = [str(x) for x in df.columns]
  # df.plot(kind='bar', yerr=df.sem(axis=0),y=df.mean(axis=0), use_index=True, ax=ax0) #
  # bar_width = 0.55
  vals = df.mean(axis=0)
  # print len(df.columns), len(vals)
  plt.bar(range(len(labels)), vals,yerr=df.sem(axis=0), width=0.55)
  plt.xticks(range(len(labels)), labels, rotation='vertical')
  save_plot_figure_2disk(plotname="/tmp/hett")
