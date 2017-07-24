import math
import re
import david as pcfg
import networkx as nx
import graph_sampler as gs
import product
import tw_karate_chop as tw
from cikm_experiments import kronfit
from gg import binarize, graph_checks, graphical_degree_sequence, grow

__authors__ = 'saguinag,tweninge,dchiang'
__contact__ = '{authors}@nd.edu'
__version__ = "0.1.0"

# infinitymirror.py  

# VersionLog:
# 0.1.0 Initial state; 


prod_rules = {}
debug = False


def learn_grammars_production_rules(input_graph):
  G = input_graph
  # print G.number_of_nodes()
  # print G.number_of_edges()
  num_nodes = G.number_of_nodes()

  G.remove_edges_from(G.selfloop_edges())
  giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
  G = nx.subgraph(G, giant_nodes)

  graph_checks(G)

  print
  print "--------------------"
  print "-Tree Decomposition-"
  print "--------------------"

  if num_nodes >= 500:
    for Gprime in gs.rwr_sample(G, 2, 100):
      T = tw.quickbb(Gprime)
      root = list(T)[0]
      T = tw.make_rooted(T, root)
      T = binarize(T)
      root = list(T)[0]
      root, children = T
      tw.new_visit(T, G, prod_rules)
  else:
    T = tw.quickbb(G)
    root = list(T)[0]
    T = tw.make_rooted(T, root)
    T = binarize(T)
    root = list(T)[0]
    root, children = T
    tw.new_visit(T, G, prod_rules)

  # return
  return prod_rules


# ~~~~~~~~~~~~~~~~
# * Main - Begin *

# seed graph
# G = nx.karate_club_graph()
G = nx.read_edgelist("../demo_graphs/as20000102.txt")  # load_graphs("KarateClub")
# gname = 'kc'
gname = 'as'

models_lst = ['phrg','chlu','kpgm']
for model in models_lst:
  print '>', model
  for i in range(10):
    #print '(m,n):', G.number_of_nodes(), G.number_of_edges()
    n = G.number_of_nodes()
    target_nodes = n
    degree_sequence = G.degree().values()

    if model == 'phrg':
      prod_rules = learn_grammars_production_rules(G)

      print
      print "--------------------"
      print "- Production Rules -"
      print "--------------------"

      for k in prod_rules.iterkeys():
        # print k
        s = 0
        for d in prod_rules[k]:
          s += prod_rules[k][d]
        for d in prod_rules[k]:
          prod_rules[k][d] = float(prod_rules[k][d]) / float(s)  # normailization step to create probs not counts.
          # print '\t -> ', d, prod_rules[k][d]
      #

      rules = []
      id = 0
      for k, v in prod_rules.iteritems():
        sid = 0
        for x in prod_rules[k]:
          rhs = re.findall("[^()]+", x)
          rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
          # print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])
          sid += 1
        id += 1

      g = pcfg.Grammar('S')
      for (id, lhs, rhs, prob) in rules:
        g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

      print "Starting max size"
      g.set_max_size(target_nodes)
      print "Done with max size"

      rule_list = g.sample(target_nodes)

      # PHRG
      pred_graph = grow(rule_list, g)[0]

    if model == 'chlu':  # ChungLu
      z = graphical_degree_sequence(target_nodes)
      pred_graph = nx.expected_degree_graph(z)


    if model == 'kpgm': # KPGM -
      k = int(math.log(target_nodes, 2))

      # from: i:/data/saguinag/Phoenix/demo_graphs/karate.txt
      # karate: P = [[0.9999,0.661],[0.661,     0.01491]]
      # Interent: autonomous systems
      # P = [[0.9523, 0.585],[0.585,     0.05644]]
      P = kronfit(G)
      print P
      pred_graph = product.kronecker_random_graph(k, P)
      for u, v in pred_graph.selfloop_edges():
        pred_graph.remove_edge(u, v)

    ofname = "../Results/{}_{}_{}.gpickle".format(gname,model,i)
    nx.write_gpickle(pred_graph, ofname)
    print ofname
    G = pred_graph

print 'Done'
# write this graph to tmp.edglst to fit Kron params
# nx.write_edgelist(G,'tmp.edglst')

#   multiGraphs.append(hstar)
#   chungluGraphs.append(cl_grph)
#   # kronGraphs.append(KPG)

#   # print "H* nodes: " + str(hstar.number_of_nodes())
#   # print "H* edges: " + str(hstar.number_of_edges())
#   # graphletG.append(gs.subgraphs_cnt(G, 100))
#   # graphletH.append(gs.subgraphs_cnt(hstar, 100))
# #

# plt.legend(labels=['Orig Graph','PHRG','Chung-Lu','KronProd'])

# plt.savefig('outfig', bb_inches='tight')
# plt.close()

# print '(m,n):', clgraph.number_of_nodes(), clgraph.number_of_edges()
