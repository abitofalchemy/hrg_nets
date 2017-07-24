import os
import sys
import re
import argparse
import traceback
import pandas as pd
import networkx as nx
import tree_decomposition as td
import net_metrics as metrics
import graph_sampler as gs
import probabilistic_cfg as pcfg

__version__ = "0.1.0"
__author__ = ['Salvador Aguinaga']

# issues: save_graph_groups has issues

def save_graph_groups(array_of_graph_objects):
  if len(array_of_graph_objects) > 3:
    return
  import shelve
  shl = shelve.open('../Results/SetOfSyntheticGraphs.dbm')
  shl['phrg'] = array_of_graph_objects[0]
  shl['clgm'] = array_of_graph_objects[1]
  shl['kpgm'] = array_of_graph_objects[2]
  shl.close()


def derive_production_rules(G):
  """

  Parameters
  ----------
  G : input graph
  """
  from PHRG import graph_checks, binarize
  prod_rules = {}

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
            T = binarize(T)
            root = list(T)[0]
            root, children = T
            td.new_visit(T, G, prod_rules)
  else:
        T = td.quickbb(G)
        root = list(T)[0]
        T = td.make_rooted(T, root)
        T = binarize(T)
        root = list(T)[0]
        root, children = T
        td.new_visit(T, G, prod_rules)

  print
  print "--------------------"
  print "- Production Rules -"
  print "--------------------"

  for k in prod_rules.iterkeys():
      print k
      s = 0
      for d in prod_rules[k]:
          s += prod_rules[k][d]
      for d in prod_rules[k]:
          prod_rules[k][d] = float(prod_rules[k][d]) / float(s)  # normailization step to create probs not counts.
          #print '\t -> ', d, prod_rules[k][d]

  return prod_rules



# def get_parser():
#   parser = argparse.ArgumentParser(description='hrgm: Hyperedge Replacement Grammars Model')
#   parser.add_argument('graph', metavar='GRAPH', help='graph path to process')
#   parser.add_argument('--version', action='version', version=__version__)
#   return parser

def load_graph():
  parser = get_parser()
  args = vars(parser.parse_args())

  if not args['graph']:
    parser.print_help()
    os._exit(1)

  try:
    g = nx.read_edgelist(args['graph'], delimiter="\t")
  except Exception, e:
    print 'Exception ERROR, UNEXPECTED SAVE PLOT EXCEPTION'
    print str(e)
  finally:
    print 'in finally'
    try:
      g = nx.read_edgelist(args['graph'], delimiter="\t", comments="%")
    except Exception, e:
      print str(e)
    finally:
      g = nx.read_edgelist(args['graph'])

  print 'Read Edgelist File'
  print '\t', g.number_of_nodes(), g.number_of_edges()
  name = os.path.basename(args['graph']).rstrip('.txt')
  return name, g

def grow_graphs_using_rules(production_rules, n=0, recrncs=1):
  from PHRG import grow

  if n == 0:
    return
  prod_rules = production_rules
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
  num_nodes = n
  g.set_max_size(num_nodes)

  print "Done with max size"

  Hstars = []

  for i in range(0, recrncs):
      rule_list = g.sample(num_nodes)
      hstar = grow(rule_list, g)[0]
      print '\tPHRG -> run:', i, str(hstar.number_of_nodes()), str(hstar.number_of_edges())
      g = hstar
  Hstars.append(hstar)

  return Hstars

def graph_generators(orig_graph, prod_rules, graph_name, nbr_rcrncs =1):
  runs_to_average = 2
  # --< PHRG >--
  phrgs_arr=[]
  for r in range(runs_to_average):
    phrg_graph = grow_graphs_using_rules(phrg_prod_rules, orig_graph.number_of_nodes(), nbr_rcrncs)
    print phrg_graph[0].number_of_nodes(), phrg_graph[0].number_of_edges()
    phrgs_arr.append(phrg_graph)

  # --< Chung-Lu >--
  clgms_arr=[]
  from salPHRG import get_clgm_graph_recurrence
  for r in range(runs_to_average):
    clgm_graph = get_clgm_graph_recurrence(orig_graph, nbr_rcrncs)
    print clgm_graph[0].number_of_nodes(), clgm_graph[0].number_of_edges()
    clgms_arr.append(clgm_graph)

  # --< Kronecker >--
  from salPHRG import grow_graphs_using_kpgm
  kpgms_arr=[]
  for r in range(runs_to_average):
    kpgm_graph = grow_graphs_using_kpgm(orig_graph, nbr_rcrncs)
    print kpgm_graph[0].number_of_nodes(), kpgm_graph[0].number_of_edges()
    kpgms_arr.append(kpgm_graph)

  return [phrgs_arr, clgms_arr, kpgms_arr]



def get_parser():
  parser = argparse.ArgumentParser(description='hrg_xperiments: Run Experiments')
  parser.add_argument('graph', metavar='GRAPH', help='graph path to process')
  parser.add_argument('--version', action='version', version=__version__)
  return parser

if __name__ == "__main__":
  global name

  reccurrence_nbr = 2
  try:
    name,graph_obj = load_graph()
    print 'Finihsed loadign the graph.'

    phrg_prod_rules = derive_production_rules(graph_obj) # learn the production rules for the given graph
    print 'Finished learning the rules.'

    [phrg_graphs, clgm_graphs, kpgm_graphs] = graph_generators(graph_obj, phrg_prod_rules, name, reccurrence_nbr)
    print len(phrg_graphs),len(clgm_graphs),len(kpgm_graphs)

    from net_metrics import save_degree_probability_distribution
    save_degree_probability_distribution(orig_g_M=[graph_obj],
                                         pHRG_M=phrg_graphs,
                                         chunglu_M=clgm_graphs,
                                         kron_M=kpgm_graphs,
                                         in_graph_str=name)

    # from net_metrics import save_eigenvector_centrality
    # save_eigenvector_centrality(orig_g_M=[graph_obj],
    #                             pHRG_M=phrg_graphs,
    #                             chunglu_M=clgm_graphs,
    #                             kron_M=kpgm_graphs,
    #                             in_graph_str=name)

    print name

  except Exception, e:
    print 'ERROR, UNEXPECTED SAVE PLOT EXCEPTION'
    print str(e)
    traceback.print_exc()
    os._exit(1)
  sys.exit(0)