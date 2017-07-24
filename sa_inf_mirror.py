import math
import re
import argparse
import traceback
import sys, os
import david as pcfg
import networkx as nx
import pandas as pd
import graph_sampler as gs
import product
import tw_karate_chop as tw
from cikm_experiments import kronfit
from gg import binarize, graph_checks, graphical_degree_sequence, grow
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

__authors__ = 'saguinag,tweninge,dchiang'
__contact__ = '{authors}@nd.edu'
__version__ = "0.1.0"

# sa_inf_mirror.py

# VersionLog:
# 0.1.0 Initial state; 


prod_rules = {}
debug = False
dbg = False

def degree_probabilility_distribution(orig_g, mGraphs, gname):
  with open('../Results/inf_deg_dist_{}_{}.txt'.format(gname,nick), 'w') as f:
    d = orig_g.degree()
    n = orig_g.number_of_nodes()
    df = pd.DataFrame.from_dict(d.items())
    gb = df.groupby([1]).count()
    gb['pk'] = gb[0]/float(n)

    f.write('# original graph degree prob distr\n')
    for row in gb['pk'].iteritems():
      f.write('({}, {})\n'.format(row[0],row[1]))

    mdf = pd.DataFrame()

    for i, hstar in enumerate(mGraphs):
      d = hstar.degree()
      df = pd.DataFrame.from_dict(d.items())
      mdf = pd.concat([mdf, df])

    mgb = mdf.groupby([1]).count()
    mgb['pk'] = mgb[0]/float(n)/float(len(mGraphs))

    f.write('# synth graph {} \n'.format(i))
    for row in mgb['pk'].iteritems():
      f.write('({}, {})\n'.format(row[0], row[1]))




def learn_grammars_production_rules(input_graph):
  G = input_graph
  # print G.number_of_nodes()
  # print G.number_of_edges()
  num_nodes = G.number_of_nodes()

  G.remove_edges_from(G.selfloop_edges())
  giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
  G = nx.subgraph(G, giant_nodes)

  graph_checks(G)

  if dbg:
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


def PHRG(G, gname):
  n = G.number_of_nodes()
  target_nodes = n
  # degree_sequence = G.degree().values()

  prod_rules = learn_grammars_production_rules(G)
  if dbg:
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

  return pred_graph


def CLGM(G, gname):
  target_nodes = G.number_of_edges()
  z = graphical_degree_sequence(target_nodes,1)
  pred_graph = nx.expected_degree_graph(z)

  return pred_graph


def KPGM(G, gname):
  target_nodes = G.number_of_nodes()
  k = int(math.log(target_nodes, 2))
  print 'k=', k

  # from: i:/data/saguinag/Phoenix/demo_graphs/karate.txt
  # karate: P = [[0.9999,0.661],[0.661,     0.01491]]
  # Interent: autonomous systems
  # P = [[0.9523, 0.585],[0.585,     0.05644]]
  P = kronfit(G)
  #print 'kronfit params (matrix):', P
  pred_graph = product.kronecker_random_graph(k, P)
  for u, v in pred_graph.selfloop_edges():
    pred_graph.remove_edge(u, v)

  return pred_graph


def main(gname_path):
  if gname_path is None: return
  print gname_path

  G = nx.read_edgelist(gname_path)

  avg_synth_graphs = []
  for i in range(10):
    synth_graph = PHRG(G, 'phrg')
    avg_synth_graphs.append(synth_graph)

  print 'performing deg dist'
  degree_probabilility_distribution(G, avg_synth_graphs, gname='phrg')

def average_10th_recursion(gname_path):
  if gname_path is None: return
  print gname_path
  orig_g = nx.read_edgelist(gname_path, comments='%')

  phrg_avg_synth_graphs = []
  kpgm_avg_synth_graphs = []
  clgm_avg_synth_graphs = []
  for run in range(10):
    print 'run >', run
    G = orig_g
    for rec in range(10):
        synth_graph = PHRG(G, 'phrg_Ten')
        G = synth_graph
    phrg_avg_synth_graphs.append(synth_graph) # takes the 10th recursion

    for rec in range(10):
        synth_graph = KPGM(G, 'kpgm_Ten')
        G = synth_graph
    # end recursion
    kpgm_avg_synth_graphs.append(synth_graph) # takes the 10th recursion

    for rec in range(10):
        synth_graph = CLGM(G, 'clgm_Ten')
        G = synth_graph
    # end recursion
    clgm_avg_synth_graphs.append(synth_graph) # takes the 10th recursion

  print 'performing deg dist'
  G = nx.read_edgelist(gname_path,comments='%')

  degree_probabilility_distribution(G, phrg_avg_synth_graphs, gname='phrg_10th')
  degree_probabilility_distribution(G, kpgm_avg_synth_graphs, gname='kpgm_10th')
  degree_probabilility_distribution(G, clgm_avg_synth_graphs, gname='clgm_10th')


def average_1st_recursion(gname_path):
  if gname_path is None: return
  print gname_path
  orig_g = nx.read_edgelist(gname_path, comments='%')

  phrg_avg_synth_graphs = []
  kpgm_avg_synth_graphs = []
  clgm_avg_synth_graphs = []
  for run in range(10):
    print '>', run
    G = orig_g
    for rec in range(10):
        synth_graph = PHRG(G, 'phrg_One')
        break
    phrg_avg_synth_graphs.append(synth_graph) # takes the 10th recursion

    for rec in range(10):
        synth_graph = KPGM(G, 'kpgm_One')
        break
    # end recursion
    kpgm_avg_synth_graphs.append(synth_graph) # takes the 10th recursion

    for rec in range(10):
        synth_graph = CLGM(G, 'clgm_One')
        break
    # end recursion
    clgm_avg_synth_graphs.append(synth_graph) # takes the 10th recursion

  print 'performing deg dist'
  G = nx.read_edgelist(gname_path, comments='%')

  degree_probabilility_distribution(G, phrg_avg_synth_graphs, gname='phrg_1st')
  degree_probabilility_distribution(G, kpgm_avg_synth_graphs, gname='kpgm_1st')
  degree_probabilility_distribution(G, clgm_avg_synth_graphs, gname='clgm_1st')

  print 'Done with: average_1st_recursion'

def get_parser():
  parser = argparse.ArgumentParser(description='sa_inf_mirror')
  parser.add_argument('graph_path', metavar='GRAPH_PATH', nargs=1, help='the graph name to process')
  parser.add_argument('nick_name', metavar='NICK_NAME', nargs=1, help='Nick name for the output file')
  parser.add_argument('--version', action='version', version=__version__)
  return parser


# ~~~~~~~~~~~~~~~~
# * Main - Begin *
if __name__ == '__main__':

  parser = get_parser()
  args = vars(parser.parse_args())
  global nick

  nick = args['nick_name'][0]
  try:
    # main(args['graph_path'][0])
    average_1st_recursion(args['graph_path'][0])
    average_10th_recursion(args['graph_path'][0])
  except Exception, e:
    print 'ERROR, UNEXPECTED SAVE PLOT EXCEPTION'
    print str(e)
    traceback.print_exc()
    os._exit(1)
  print 'Done'
  sys.exit(0)


