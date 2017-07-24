__author__ = 'tweninge'+'@'+'nd.edu'
__author__ = 'saguinag'+'@'+'nd.edu'
__author__ = 'rodrigopala91'+'@'+'gmail.com'
__version__ = "0.1.0"

##
##  hrgm = hyperedge replacement grammars model
##

# TODO: Sort out the returns
# VersionLog:
# 0.0.1 Initial state; modified tw_karate_chop to accommodate this version
#

import argparse,traceback,optparse
import pandas as pd
import numpy as np
import os, sys, time
import networkx as nx
from collections import deque, defaultdict, Counter
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import itertools
import tw_karate_chop as tw

graphs_list=['BoardEx','KarateClub','NewWatStrSW','Erdos-Renyi','Kronecker']
global num_nodes, debug, prod_rules
prod_rules={}
debug = False

##
## [ david
def partitions(n, k):
    """Generate k-tuples of positive integers that sum to n."""
    if k == n == 0:
        yield ()
        return
    assert k >= 1
    assert n >= 1
    for comb in itertools.combinations(range(1,n), k-1):
        part = []
        prev = 0
        for cur in comb:
            part.append(cur-prev)
            prev = cur
        part.append(n-prev)
        yield tuple(part)

class Rule(object):
    def __init__(self, id, lhs, rhs, prob):
        self.id = id
        self.lhs = lhs
        self.rhs = rhs
        self.prob = prob

class Grammar(object):
    def __init__(self, start):
        self.start = start
        self.nonterminals = {start}
        self.by_lhs = {}

    def add_rule(self, rule):
        self.nonterminals.add(rule.lhs)
        self.by_lhs.setdefault(rule.lhs, []).append(rule)

    def generate_rules(self, lhs, size):
        for rule in g.by_lhs[lhs]:
            rhs = rule.rhs
            #need to parse differently SAL
            rhs_nonterminals = [x for x in rhs if x in self.nonterminals]
            rhs_terminals = [x for x in rhs if x not in self.nonterminals]

            # Every nonterminal has size >= 1, so skip impossible cases
            if len(rhs) > size or len(rhs_nonterminals) == 0 and len(rhs) < size:
                continue

            for rhs_nonterminal_sizes in partitions(size-len(rhs_terminals), len(rhs_nonterminals)):
                rhs_sized = []
                i = 0
                for y in rhs:
                    if y in self.nonterminals:
                        rhs_sized.append((y, rhs_nonterminal_sizes[i]))
                        i += 1
                    else:
                        rhs_sized.append((y, 1))
                yield Rule(rule.id, lhs, rhs_sized, rule.prob)

    def topological_sort(self):
        # Topologically sort the unary rules

        dep = {x:set() for x in self.nonterminals}
        for lhs, rules in self.by_lhs.items():
            for rule in rules:
                if len(rule.rhs) == 1 and rule.rhs[0] in self.nonterminals:
                    dep[rule.lhs].add(rule.rhs[0])
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
        return topological

    def set_max_size(g, max_size):
        topological = g.topological_sort()
        alpha = {}
        for size in range(1, max_size+1):
            for lhs in topological:
                for rule in g.generate_rules(lhs, size):
                    p = np.log(rule.prob)
                    try:
                        for y, ysize in rule.rhs:
                            if y not in g.nonterminals:
                                pass
                            elif (y, ysize) in alpha:
                                p += alpha[y, ysize]
                            else:
                                raise ZeroProbability()
                    except ZeroProbability:
                        continue

                    if (lhs, size) in alpha:
                        alpha[lhs, size] = np.logaddexp(alpha[lhs, size], p)
                    else:
                        alpha[lhs, size] = p
        g.alpha = alpha

    def sample(g, size):
        w = [(g.start, size)]
        i = 0
        rules = []
        while i < len(w):
            if w[i][0] in g.nonterminals:
                lhs, lhs_size = w[i]
                choices = []
                for rule in g.generate_rules(lhs, lhs_size):
                    p = np.log(rule.prob)
                    try:
                        for y, ysize in rule.rhs:
                            if y not in g.nonterminals:
                                pass
                            elif (y, ysize) in g.alpha:
                                p += g.alpha[y, ysize]
                            else:
                                raise ZeroProbability()
                    except ZeroProbability:
                        continue
                    p = np.exp(p-g.alpha[lhs, lhs_size])
                    choices.append((p, rule))
                z = sum(p for p, _ in choices) # should be close to 1
                r = np.random.uniform(0, z, ())
                s = 0.
                for p, rule in choices:
                    s += p
                    if s > r:
                        break
                else:
                    assert False
                rules.append(rule.id)
                w[i:i+1] = rule.rhs
            else:
                i += 1
        return rules

class ZeroProbability(Exception):
    pass
## ends david]
##


class ListSupportedGraphs(argparse.Action):
  def __call__(self, parser, args, values, option_string=None):
    ##
    print ">> GRAPH_NAME: list of supported graphs"
    for g in graphs_list:
      print "  '%s'" % g

def write_to_json(data, model_name, graphs=False):
  import json, time
  timestr = time.strftime("%d%b%y_%H%M%S")
  model_nm = "/tmp/"+model_name.replace (" ", "_")
  if save_unique_names_bool:
    out_file = model_nm+'_'+timestr
  else:
    out_file = model_nm

  # TODO: are the next 2 lines needed?
  #if os.path.isfile(out_file):
  #  out_file = "/tmp/"+model_name+'_'+timestr+'_1.json'

  try:
    if graphs:  ## when writing graphs to a file
      for g in data:
        #print type (g)
        df = pd.DataFrame(g.edges_iter())
        df.columns=['% src_node','trg_node']
        out_filename = out_file+"_edgeList_grph.tsv"
        df.to_csv(out_filename, mode='a',sep='\t', index=False, encoding='utf-8', header=True)
      print '  Graphs saved to edge-list file:',out_filename
    else:
      out_filename = out_file+".json"
      with open(out_filename, 'w') as fp:
          json.dump(data, fp)
      print '\t',out_filename,'file saved.'
  except Exception, e:
    print 'ERROR, UNEXPECTED EXCEPTION'
    print str(e)
    traceback.print_exc()
    os._exit(1)

def load_graph(graph_name):
  if graph_name=='':
    return
  if graph_name == 'BoardEx':
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
  elif graph_name == 'KarateClub':
    G = nx.karate_club_graph()
  elif graph_name == "Erdos-Renyi":
    n=68 # 10 nodes
    m=68 # 20 edges
    G=nx.gnm_random_graph(n,m) #binomial_graph which is also somtims called the Erdos-Renyi grap    h.
  elif graph_name == "NewWatStrSW":
    k=2
    p=0.5
    G = nx.newman_watts_strogatz_graph(n, k, p)
  elif graph_name == "Kronecker":
    ## TODO: implement Kronecker
    G = nx.graph()
    #
  else:
    G = nx.graph()

  return (graph_name,G)

def load_graphs(args):
  print '-'*100
  print '... loading graph:',args['graph_name']
  #hrgs = HyperEdgeRGs(args['graph_name'])
  #print hrgs.graph_name
  avail = False
  for g in graphs_list:
    if g == args['graph_name'][0]:
      G = load_graph(g)
      ## Check graph
      # print G[0]
      #print "%s -> G(n=%d,m=%d)" % (G.number_of_nodes(), G.number_of_edges()
      avail = True
      return G
  if not avail:
    print '!!Warining: graph is not available'
    os._exit(1)


def graph_checks(G):
  ## Target number of nodes
  global num_nodes
  num_nodes = G.number_of_nodes()

  if not nx.is_connected(G) :
    print "Graph must be connected";
    os._exit(1)

  if G.number_of_selfloops() > 0 :
    print "Graph must be not contain self-loops";
    os._exit(1)


def get_parser():
    parser = argparse.ArgumentParser(description='hrgm: Hyperedge Replacement Grammars Model')
    parser.add_argument('graph_name', metavar='GRAPH_NAME', nargs=1, help='the graph name to process')
    parser.add_argument('--list',  nargs=0, help='unique file names', action=ListSupportedGraphs)
    parser.add_argument('-s','--save',  help='Save to disk with unique names', action='store_true', default=False)
    parser.add_argument('--version', action='version', version=__version__)
    return parser

if __name__ == '__main__':
  #global options, args
  #g = command_line_runner()
  parser = get_parser()
  args = vars(parser.parse_args())
  global save_unique_names_bool
  save_unique_names_bool = args['save']

  if not args['graph_name']:
        parser.print_help()
        os._exit(1)

  (gn,G) = load_graphs(args)
  graph_checks(G)



  T = tw.quickbb(G)      # tree decomposition
  root = list(T)[0]

  print
  print "---------------------"
  print "- Intersection Tree -"
  print "---------------------"

  #get S
  rhs = tw.get_production_rule(G, root, set())
  s = list()
  for c in T[root]:
    s.append( list(set(c).intersection(root)) )
  tw.add_to_prod_rules(prod_rules, set(), rhs, s)
  tw.visit(root, 0, set(),prod_rules, T, G)

  print
  print "--------------------"
  print "- Production Rules -"
  print "--------------------"

  for k in prod_rules.iterkeys():
    print k
    for d in prod_rules[k]:
        print '\t -> ', d, prod_rules[k][d]

  # print type(prod_rules),  len(prod_rules)
  # print prod_rules

  print
  print "--------------------------"
  print "- Grammar Data Strcuture -"
  print "--------------------------"
  ## [David's new sampler]
  """
  The maximum size you want to sample: set_max_size
  sample(n): sample a string of length n
  """
  # global g
  g = Grammar('(S)')

  id = 0
  for lhs,rhss in prod_rules.iteritems():
      for x in rhss.keys():
          g.add_rule(Rule(id, lhs, x, rhss[x]))
  
  nbrsamps = 8
  g.set_max_size(nbrsamps) ## sets the maximum size you will wanto sample in the future
  #print g.sample(1)

  for i in range(1000):
     print g.sample(nbrsamps)
""""

  for (id, lhs, rhs, prob) in rules:
      g.add_rule(Rule(id, lhs, rhs, prob))

  print g.items()
  exit()
  g.set_max_size(15)
  for i in range(1000):
      print g.sample(15)
"""
  # write_to_json(prod_rules, gn) ## write production rules to disk
  #
  # n_distribution = {}
  #
  # eff_diag_run = pd.DataFrame()
  # grphs = []
  #
  # print
  # print "--------------------"
  # print "- Runs             -"
  # print "--------------------"
  #
  # compute_eff_diameter = False
  # heterm_s = [] # Stats
  # nbr_of_runs = 100
  # for run in range(0,nbr_of_runs):
  #     H = list()
  #     N = list()
  #     heterm_cnt = Counter()
  #
  #     N.append(["S"]) #starting node
  #     ttt=0
  #
  #     eff_dia_gph = []
  #     graphlet_node_cnt = []
  #
  #     #pick non terminal
  #     num = 0
  #     while len(N) > 0:
  #         lhs_match = find_match(N, prod_rules)
  #         e = []
  #         match = []
  #         for tup in lhs_match:
  #             match.append(tup[0])
  #             e.append(tup[1])
  #         lhs_str = "(" + ",".join(str(x) for x in sorted(e)) + ")"
  #         #DO SOMETHING USEFUL WITH THIS MATCH
  #         new_idx = {}
  #         n_rhs =str(control_rod(H, num_nodes, prod_rules[lhs_str].items())).lstrip("(").rstrip(")")
  #         #n_rhs =str(weighted_choice(prod_rules[lhs_str].items())).lstrip("(").rstrip(")")
  #         #n_rhs = str(random.choice(prod_rules[lhs_str].keys())).lstrip("(").rstrip(")")
  #         # if n_rhs[-1] == "N":
  #         if debug: print lhs_str, "->", n_rhs
  #
  #         for x in n_rhs.split(")("):
  #             heterm_cnt[x] += 1
  #             new_he = []
  #             he = x.split(":")[0]
  #             term_symb = x.split(":")[1]
  #
  #             for y in he.split(","):
  #                 if y.isdigit(): # y is internal node
  #                     if y not in new_idx:
  #                         new_idx[y] = num
  #                         num+=1
  #                     new_he.append(new_idx[y])
  #                 else: #y is external node
  #                     for tup in lhs_match: #which external node?
  #                         if tup[1] == y:
  #                             new_he.append(tup[0])
  #                             break
  #             #prod = "(" + ",".join(str(x) for x in new_he) + ")"
  #             if term_symb == "N":
  #                 N.append(sorted(new_he))
  #             elif term_symb == "T":
  #                 H.append(new_he)
  #         ## ends for
  #         ##print 'N:',len(N), 'H:',len(H)
  #
  #         match = sorted(match)
  #         N.remove(match)
  #
  #         if compute_eff_diameter:
  #           ##
  #           ## EFFECTIVE DIAMETER as graph grows
  #           ##
  #           newG = nx.Graph()
  #           for e in H:
  #             if (len(e) == 1):
  #               newG.add_node(e[0])
  #             else:
  #               newG.add_edge(e[0], e[1])
  #           #eff_dia_gph.append(chartvis.bfs_eff_diam(newG, 50, .90))
  #           #eff_dia_gph.append((newG.number_of_nodes(), chartvis.bfs_eff_diam(newG, 50, .90)))
  #           eff_dia_gph.append([newG.number_of_nodes(),
  #                             chartvis.bfs_eff_diam(newG, 50, .90)])
  #     if compute_eff_diameter:
  #       ed = map(list, zip(eff_dia_gph))
  #       eff_diag_run = eff_diag_run.append(pd.DataFrame(eff_dia_gph))
  # ######
  #
  #     newG = nx.Graph()
  #     for e in H:
  #         if(len(e) == 1):
  #             newG.add_node(e[0])
  #         else:
  #             newG.add_edge(e[0], e[1])
  #
  #     #print len(newG.edges())
  #
  #     n = newG.number_of_nodes()
  #     if n in n_distribution:
  #         n_distribution[newG.number_of_nodes()] += 1
  #     else:
  #         n_distribution[newG.number_of_nodes()] = 1
  #
  #     #if n == num_nodes:
  #     grphs.append(newG)
  #
  #     heterm_s.append(heterm_cnt) ## keep each run's count of HETT
  #
  #     # print run number
  #     if debug: print '\trun:',run,'in',nbr_of_runs
  # #------------------------ for loop ends
  #
  # if compute_eff_diameter:
  #   print eff_diag_run.shape
  #   print eff_diag_run.head()
  #   ef =  eff_diag_run.groupby([0])
  #
  # # print "V = ", newG.number_of_nodes()
  # #      print "E = ", newG.number_of_edges()
  # #      giant_nodes = max(nx.connected_component_subgraphs(newG), key=len)
  # #      giant = nx.subgraph(newG, giant_nodes)
  # #      print "V in giant component = ", giant.number_of_nodes()
  # #      print "E in giant compenent = ", giant.number_of_edges()
  # #      print "Diameter = ", nx.diameter(nx.subgraph(newG, giant))
  #
  #
  # ## Print the distribution
  # x =[];  y =[]
  # for k in sorted(n_distribution.keys()):
  #   print k,"\t",n_distribution[k]
  #   x.append(k)
  #   y.append(n_distribution[k])
  #
  # ## Save Graphs to Disk
  # write_to_json(grphs,gn,graphs=True)
  #
  # ## Chartvis
  # print '-'*80
  #
  # print args


    # g = command_line_runner()

    # ## View/Plot the graph to a file
    # fig = plt.figure(figsize=(1.6*6,1*6))
    # ax0 = fig.add_subplot(111)

    # nx.draw_networkx(g[1],ax=ax0)
    # plt_filename="/tmp/outfig"

    # try:
    #   save_plot_figure_2disk(plotname=plt_filename)
    #   print 'Saved plot to: '+plt_filename
    # except Exception, e:
    #   print 'ERROR, UNEXPECTED SAVE PLOT EXCEPTION'
    #   print str(e)
    #   traceback.print_exc()
    #   os._exit(1)
    # sys.exit(0)
    #main()
    #sys.exit(0)
