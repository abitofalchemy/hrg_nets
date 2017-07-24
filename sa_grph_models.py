#!/Users/saguinag/ToolSet/anaconda/bin/python
# -*- coding: utf-8 -*-
# simp_tst

__author__ = "Sal Aguinaga"
__copyright__ = "Copyright 2015, The Phoenix Project"
__credits__ = ["Sal Aguinaga", "Rodrigo Palacios", "Tim Weninger"]
__license__ = "GPL"
__version__ = "0.1.0"
__maintainer__ = "Sal Aguinaga"
__email__ = "saguinag (at) nd dot edu"
__status__ = "sa_grph_models"

from phoenix import phoenix, helpers, putils, charvis
from pprint import PrettyPrinter as pp
from os import chdir
pp = pp(2).pprint
import networkx as nx
import datetime, timeit
import pickle
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

##########
# @author: jonathanfriedman
#
def jsd(x,y): #Jensen-shannon divergence
  import warnings
  warnings.filterwarnings("ignore", category = RuntimeWarning)
  x = np.array(x)
  y = np.array(y)
  d1 = x*np.log2(2*x/(x+y))
  d2 = y*np.log2(2*y/(x+y))
  d1[np.isnan(d1)] = 0
  d2[np.isnan(d2)] = 0
  d = 0.5*np.sum(d1+d2)
  return d

#jsd(np.array([0.5,0.5,0]),np.array([0,0.1,0.9]))

## For timing info
def hms_string(sec_elapsed):
  h = int(sec_elapsed / (60 * 60))
  m = int((sec_elapsed % (60 * 60)) / 60)
  s = sec_elapsed % 60.
  return "{}:{:>02}:{:>05.2f}".format(h, m, s)
# End hms_string

def load_cliques_obj(filename):
  #pickleFileName = "/data/saguinag/datasets/arxiv/cond_mat_03_cliques.pkl"
  pickleFileName = "../demo_graphs/"+filename+".pkl"
  t0 = datetime.datetime.now()
  with open(pickleFileName, 'rb') as ifile:
      data = pickle.load(ifile)
    #print(datetime.datetime.now()-t0,' elapsed time.')
  return data

def getCliques(g, filename):
  netscience_graph = g
  t0 = datetime.datetime.now()
  cliques = list(nx.find_cliques(netscience_graph))
    #print(datetime.datetime.now()-t0,' elapsed time.')
  pickleFileName = "../demo_graphs/"+filename+".pkl"
  with open(pickleFileName, 'wr') as outf:
    pickle.dump(cliques, outf, protocol=pickle.HIGHEST_PROTOCOL)
  return cliques

def load_graph(filename):
  t0 = datetime.datetime.now()
  #netscience_graph = nx.read_gml('/data/saguinag/datasets/arxiv/cond-mat-2003.gml')
  if (filename is None):
    if debug: print('! Error no filename')
    return None
  print('Loading ' + filename)
  graph = nx.read_gml('../demo_graphs/'+filename)
  #print(datetime.datetime.now()-t0,' elapsed time.')
  return (graph)

def cliques_exists_for(filename):
  from os import path
  pickleFileName = "../demo_graphs/"+filename+".pkl"
  print('Looking for '+pickleFileName)
  if path.exists(pickleFileName):
    return True
  else :
    return False

############--------------------------------------------------------------------

## function find_clique_pairs_with_intersection_size
def find_clique_pairs_with_intersection_size(clique_size, clq_x_clq_model):
    return [ky for ky in clq_x_clq_model.keys() if clique_size in ky]

#pp(clqs_x_clqs_dist)
# tst: find_clique_pairs_with_intersection_size(3,clqs_x_clqs_dist)
def intxn_size_from_clq_pair_dist(clique_size, clq_x_clq_model):
    #print "intxn_size_from_clq_pair_dist"
    pairs = [ky for ky in clq_x_clq_model.keys() if clique_size in ky]

    return pairs

def sel_intxn_sz_from(pairs, dist):
    ## returns an intersection size
    import pandas as pd
    dist_sets=  [dist[pr] for pr in pairs]
    #print dist_sets
    df = pd.DataFrame.from_dict([v for v in dist_sets])
    #print df
    mean_dist = df[:].mean()
    t_val = sample(df[:].mean(),1)
    return mean_dist[mean_dist == t_val].index[0]

def combine_sets(clique, total_nodes):
  #print clique,total_nodes
  if len(list(clique)) == total_nodes:
    return clique ## return the tuple
  else:
    ## take out the clique.nodes and then add combinations
    nodes = range(0,total_nodes)
    return list(combinations(nodes, 2))

def fuse_cliques_pair(clq_pr, kl_threshold, cliq_numb_dist, poss_combs):
  from heapq import heappush, heappop
  
  print("-- fusing the nodes ...")
  # poss_combs has the size of the intersection embedded
  # clq_pr tells us the the left and right side cliques.
  print "-- left and right sides", clq_pr
  jsdh = []#jsd_dict_heap = dict()
  
  ## test combinations for minimal divergence from reference
  for intrsct_nodes in poss_combs:
    #print "  ", intrsct_nodes
    edg_lst  = []
    lft_side = combine_sets(intrsct_nodes, clq_pr[0])
    lft_side = list(combinations(lft_side,2))
    rgt_side = combine_sets(intrsct_nodes, clq_pr[1])
    rgt_side = list(combinations(rgt_side,2))
    #print lft_side, rgt_side
    
    g = nx.Graph()
    g.add_edges_from(lft_side)
    g.add_edges_from(rgt_side)


    ## now that we created this graph, we get compress it
    ## get its distribution models
    ## and compare them to one of the input distributions
    cliques = list(nx.find_cliques(g))
    loc_cliques_dist = helpers.clique_number_distribution(cliques)
    loc_cliques_dist = helpers.normalize_distribution(loc_cliques_dist)
    loc_clq_len =  len(loc_cliques_dist.values())
    loc_cliques_dist_v = loc_cliques_dist.values()
    
    # pad the array of values with np.nan
    [loc_cliques_dist_v.append(np.nan  * x) for x in range(0,len(cliq_numb_dist.values()) - loc_clq_len)]
    heappush(jsdh, (jsd(cliq_numb_dist.values(), loc_cliques_dist_v), intrsct_nodes))
  
  min_div = heappop(jsdh)
  if debug: print min_div[1]
  ## nodes tuple
  #print list(min_div[1])
  
  ## intersection comb that yields the smallest divergence is obtained via
  ## heappop( jsdh )
  ## Given this, we freeze the graph @ this intersection
  edg_lst = []
  lft_side = combine_sets(min_div[1], clq_pr[0])
  #lft_side = list(combinations(lft_side,2))
  rgt_side = combine_sets(min_div[1], clq_pr[1])

  if debug: print lft_side, rgt_side

  g = nx.Graph()
  if (len(lft_side) == 2):
    g.add_edges_from([lft_side])
  else:
    g.add_edges_from(lft_side)
  if (len(rgt_side) == 2):
    g.add_edges_from([rgt_side])
  else:
    g.add_edges_from(rgt_side)

  return g



from random import sample
from itertools import combinations



def get_edges_from_clique(nodes):
    from itertools import combinations
    return list(combinations(nodes, 2))

def decompressS(clq_num_dist, clq_x_clq_dist, kl_threshold, iteration_cap=100, debug=False):
    ## init step
    clqs = []
    init_success = 1
    graph = nx.Graph()
    
    if debug: print "Begin..."
    while init_success:
        ## get clique size @ random
        clq_sz_atrnd = helpers.get_weighted_random_value(clq_num_dist) ## get a clique size at random
        if debug: print ("clq size:",clq_sz_atrnd)

        ## sample a intersecting size from all pair distributions that have a clique of given size
        clq_pairs = intxn_size_from_clq_pair_dist(clq_sz_atrnd,clq_x_clq_dist)
        intxn_sz = sel_intxn_sz_from(clq_pairs, clq_x_clq_dist)
        if debug: print ("intxn_sz",intxn_sz)

        ## clique pairs with a given intersection size
        clq_pairs_arr = find_clique_pairs_with_intersection_size(intxn_sz, clq_x_clq_dist)
        pp(clq_pairs_arr)
        if len(clq_pairs_arr)<1:
            if debug: print 'Trying again'
            continue
        else:
            init_success = 0
    if debug: print "Init phase done..."
    #if debug:print "clq_pairs_arr:" ,clq_pairs_arr
    ## pick clique pair at random to start with
    clq_pair = sample(clq_pairs_arr,1)[0]
    clqs.append(clq_pair[0])## ?

    ## The combination of intersections that gives the minimal kl_div is?
    nodes = range(0,max(clq_pair))
    #nodes = [nodes, range(len(nodes),clq_pair[1])]
    #nodes = [item for sublist in nodes for item in sublist]
    #print "pair: ", clq_pair, "intx sz:", intxn_sz
    #print "nodes:", nodes

    ## Possible intersecting combinations
    poss_intxns_for_given_size = list (combinations(nodes,intxn_sz))
    #print("possible combination of size |intxn_sz|", poss_intxns_for_given_size)
    #print type(poss_intxns_for_given_size)

    ## Fuse cliques along the intersection combination with minimal js_div
    graph = fuse_cliques_pair(clq_pair,
                              kl_threshold,
                              clq_num_dist, poss_intxns_for_given_size)

#nx.draw_networkx(graph)
#    plt.savefig( 'fig0.png' )
    return


def decompressA(clq_num_dist, clq_x_clq_dist, node_count, kl_threshold, iteration_cap=100, debug=False):

    """In the greedy step (decision process), we choose the operation
    that most minimizes the kl divergence."""
    nodes = list(range(node_count))
    if debug: print 'nodes',nodes
    ### Initialize clqs with a clique of random selected size
    clqs = []
    clq_size  = get_weighted_random_value(clq_num_dist)
    clq_nodes = sample(nodes, clq_size)

    clqs.append(clq_nodes)

    ### Initialize hypergraph's distributions
    ref_clq_num_dist   = normalize_distribution(clique_number_distribution(clqs))
    ref_clq_x_clq_dist = normalize_distributions(cliques_x_cliques_distribution(clqs))

    if debug: print ref_clq_num_dist, ref_clq_x_clq_dist
    ### Initialize kl divergence scores
    clq_num_kl = 100
    clq_x_clq_kl = 100
    avg_kl = (clq_num_kl+clq_x_clq_kl)/2
    ###
    i = 0

    while avg_kl > kl_threshold and i < iteration_cap:

        # climb down from arbitrarily high KL value
        min_clq_num_kl = 100
        min_clq_x_clq_kl = 100
        min_clq = []

        ### Greedy Decision Process (GDP) ###
        for r in range(1, node_count+1):
            # not-fully exhaustive; exhaustive becomes O(N!)
            random_clq_of_size_r = sample(list(combinations(nodes, r)), 1)
            if debug: print "random_clq_of_size_r",random_clq_of_size_r
            assert len(random_clq_of_size_r[0]) > 0
            continue
            # update dists to represent one step forward
            new_clq_num_dist = normalize_distribution(clique_number_distribution(clqs+random_clq_of_size_r))
            new_clq_x_clq_dist = normalize_distributions(cliques_x_cliques_distribution(clqs+random_clq_of_size_r))



            # calculate the kl divergences of next step
            kl_1 = putils.kl_divergence(clq_num_dist, new_clq_num_dist)
            kl_2 = putils.avg_kl_divergence(clq_x_clq_dist, new_clq_x_clq_dist)

            ## king of the valley(?)-type of minimization
            if (kl_1 < min_clq_num_kl and
                kl_2 < min_clq_x_clq_kl):

                min_clq_num_kl = kl_1
                min_clq_x_clq_kl = kl_2
                min_clq = random_clq_of_size_r
        ######### End of GDP ########
        ### Update list of cliques, reference models, and average KL score
        if min_clq and min_clq[0] not in clqs:
            clqs.append(list(min_clq[0]))

            ref_clq_num_dist = normalize_distribution(clique_number_distribution(clqs))
            ref_clq_x_clq_dist = normalize_distributions(cliques_x_cliques_distribution(clqs))

            clq_num_kl = min_clq_num_kl
            clq_x_clq_kl = min_clq_x_clq_kl

            assert clq_num_kl   == putils.kl_divergence(clq_num_dist, ref_clq_num_dist)
            assert clq_x_clq_kl == putils.avg_kl_divergence(clq_x_clq_dist, ref_clq_x_clq_dist)

            avg_kl = (clq_num_kl+clq_x_clq_kl)/2

            if debug:
                if debug: print("%d | %.6g | %.6g" % (i, clq_num_kl, clq_x_clq_kl))
                #print()
        i += 1

    return clqs

#### #### ####
def anneal_bycliquesweep_toGraph(l_graph, k, div, intx_cliq_freq, clq_numb_dist, debug=False):
  from collections import defaultdict
  from itertools import combinations
  from phoenix import divtools
  from heapq import heappush, heappop
  from networkx.algorithms.approximation import clique


  ## initialize
  q_min_div_4graph = []
  div_dict = dict()
  cgraph = l_graph
  
  if len(cgraph.nodes()) is 0:         ## g is null
    return putils.graph_from_clique(k) ## from Null graph to graph of size k
  """
  Below models when we sweep across all cliques in the forming graph
  
  """
  ## find the cliques in the current graph
  cliques = list(nx.find_cliques(cgraph))
  ## list of G.clique lengths
  c_lengths = [len(c) for c in cliques]
  if debug: print c_lengths, k, ':[G.c],k'

  for cl in c_lengths:
    pair = tuple((cl, k))
    if not (pair in intx_cliq_freq.keys()):
      #pair = tuple((cl, k))
      continue
    
    intxn   = helpers.get_weighted_random_value(intx_cliq_freq[pair])
    print '  :',cgraph.number_of_nodes(), pair,intxn
    if not intxn:
      print ' No intersection'
      continue
    elif debug: print " %d, %s, %s : (k,intxn,pair)" % (k, intxn, pair)
#=======
#    #print kpairs
#    j = 0
#    for pair in kpairs:
#      if not [c for c in c_lengths if set((c,k)) == set(pair)]:
#        continue 
#
#      intxn   = helpers.get_weighted_random_value(intx_cliq_freq[pair])
#      if not intxn:
#        print ' No intersection'
#        continue
#      elif not debug: print " ",intxn,':intxn', pair,':pair'

    m_nodes = len(cgraph.nodes()) + k - intxn ## using len | or max + 1

    if debug: print " ",m_nodes,':max_nodes'
    ## 
    clq_intx = set(pair).intersection(set(c_lengths))
    if clq_intx: 
      if debug: print ' ', clq_intx, ':clq_intx' 
    else:
      continue
  
    c1 = cliques[c_lengths.index(clq_intx.pop())]
    ## the clique we will anneal the new k-clique to
    if not debug: print ' ',c1,': G.clique' #
  
    base_clique_comb = set(combinations(c1,intxn))
    for c_comb in base_clique_comb:
      clq_nodes = set(c1).difference(c_comb)
      #print ' ',c_comb, clq_nodes.pop()
      #
      clq2add =[]
      clq2add.append(list(c_comb))
      clq2add.append(range(max(cgraph.nodes())+1,m_nodes))
      #print ' ',list(c_comb), range(max(cgraph.nodes())+1,m_nodes)
      #print clq2add
      clq2add =  reduce(lambda x,y: x+y,clq2add)
      if not debug: print ' ', clq2add,":clique to add"
      edge_lst = set(combinations(clq2add,2))
      if not edge_lst:
        #print '. empty edge list'
        continue
      
      if not debug: print ' ', edge_lst,':edge list set'
      cgraph.add_edges_from(edge_lst)
      
      ## get current distributions for cgraph
      clqs = list(nx.find_cliques(cgraph))
      clq_dist = helpers.clique_number_distribution(clqs)
      clq2clq_intxn_dist = phoenix.clique2cliqueIntersection(clqs)
      #
      clq_dist = helpers.normalize_distribution(clq_dist)
      clq2clq_intxn_dist = helpers.normalize_distributions(clq2clq_intxn_dist)
      c_div = divtools.jsd2(clq_numb_dist, clq_dist)
      x_div = divtools.avg_jsd2(intx_cliq_freq, clq2clq_intxn_dist)
      avg_div = np.mean([c_div, x_div]) ## regular avg.
      
      ## add avg_div to dict
      div_dict[avg_div] = edge_lst
      if debug: print ' ', avg_div,':avg div'
    # ...... ends for each combination
    
    if not len(div_dict):
      print 'no div dict'
      return cgraph
    print ' ', min(div_dict.keys()),':min div for last comb set'
  # ends for cl in G

  if len(div_dict) == 0:
    return cgraph
  print ' ', min(div_dict)
  #print ' div:',div_dict[min(div_dict)]
  edge_list = div_dict[ min(div_dict) ]
  if debug: print edge_list

  l_graph.add_edges_from(edge_list)
  #print l_graph.nodes()
  return l_graph


def anneal_clique_2best_match(l_graph, k, div, intx_cliq_freq, clq_numb_dist, debug=False):
  from collections import defaultdict
  from itertools import combinations
  from phoenix import divtools
  from heapq import heappush, heappop
  
  ## initialize
  q_min_div_4graph = []
  div_dict = dict()
  cgraph = l_graph
  edge_list = []
  
  if len(cgraph.nodes()) is 0:         ## g is null
    return putils.graph_from_clique(k) ## from Null graph to graph of size k

  ## find the cliques in the current graph
  cliques   = list(nx.find_cliques(cgraph))
  print cliques
  c_lengths = [len(c) for c in cliques]

  csubset   = [c for c in c_lengths if tuple(sorted((c,k))) in intx_cliq_freq.keys()]
  if not csubset:
    return l_graph
  print csubset

  for c in csubset:
    #print (c,k)
    pair = tuple(sorted((c,k)))
    intxn = helpers.get_weighted_random_value(intx_cliq_freq[pair])
    m_nodes = max(cgraph.nodes()) + 1 + k - intxn ## using len | or max + 1

    if not debug: print " ",m_nodes,':max_nodes'

    clq_intx = set(pair).intersection(set(c_lengths))
    if clq_intx:
      if debug: print ' ', clq_intx, ':clq_intx'
    else:
      continue

    c1 = cliques[c_lengths.index(clq_intx.pop())]
    if debug: print ' ',c1,': G.clique' #clique to anneal new k-clique to
    if not debug: print '  clique-sum(',k,').to(',c1,').along-intxn:',intxn

    newnodes= range(m_nodes - k + intxn, m_nodes)

    base_clique_comb = set(combinations(c1,intxn))
    for c_comb in base_clique_comb:
      clq_nodes = set(c1).difference(c_comb)
      #print ' ',c_comb,newnodes

      clq2add =[list(c_comb),newnodes]
#      clq2add.append(list(c_comb))
#      clq2add.append(range(max(cgraph.nodes())+1,m_nodes))
      #print ' ',list(c_comb), range(max(cgraph.nodes())+1,m_nodes)
      #print clq2add
      clq2add =  reduce(lambda x,y: x+y,clq2add)
      if not debug: print ' ',c1,'+',clq2add #,":clique to add"

      edge_lst = set(combinations(clq2add,2))
      if not edge_lst:
        #print '. empty edge list'
        continue
      
      if debug: print ' ', edge_lst,':edge list set'
      cgraph.add_edges_from(edge_lst)
      #print ' ', cgraph.number_of_nodes()

      ## get current distributions for cgraph
      clqs = list(nx.find_cliques(cgraph))
      clq_dist = helpers.clique_number_distribution(clqs)
      clq2clq_intxn_dist = phoenix.clique2cliqueIntersection(clqs)
      #
      clq_dist = helpers.normalize_distribution(clq_dist)
      clq2clq_intxn_dist = helpers.normalize_distributions(clq2clq_intxn_dist)
      c_div = divtools.jsd2(clq_numb_dist, clq_dist)
      x_div = divtools.avg_jsd2(intx_cliq_freq, clq2clq_intxn_dist)
      avg_div = np.mean([c_div, x_div]) ## regular avg.
      
      ## add avg_div to dict
      div_dict[avg_div] = edge_lst
      if debug: print ' ', avg_div,':avg div'
    # ...... ends for each combination
  

  if len(div_dict) == 0:
    return l_graph

  #print ' div:',div_dict[min(div_dict)]
  edge_list = div_dict[ min(div_dict) ]
  print ' ', min(div_dict)#, k, edge_list

  l_graph.add_edges_from(edge_list)
  #print l_graph.nodes()
  return l_graph



####

def graph_formation(clique_freq, intxn_freq, div_thr, N, debug=True):
  """
    Check the graph as we test each combination
    soemthing isn't right
    """
  node_count = 0 ## begin initialization
  g = nx.Graph() 
  j = 0 # interation count
  ## done with initialization
  print 'j : M k | j = iteration, M= number of nodes, k= clique size'
  while node_count < N:
    #print j, node_count, 
    ## get a clique at rnd from the distribution
    seed_clique = phoenix.get_weighted_random_value(clique_freq)
    seed_clique = sample(clique_freq,1)
    k_clq = seed_clique[0]
    print j,":", node_count, k_clq
    print '~'*80

    if node_count < 1:
      #if debug: print ' first iteration'
      g.add_edges_from(list(combinations(range(0,k_clq),2)))
      node_count = g.number_of_nodes()
      j = j + 1 # iteration counter
      next
    else:
      ## anneal to graph, k-clique, intxn_size
      g = anneal_clique_2best_match(g, k_clq, div_thr, intxn_freq, clique_freq)
      node_count = g.number_of_nodes()
     
      j = j + 1 # iteration counter
  
    if j >= 12: break
  return g

if __name__ == '__main__':
  from phoenix import divtools
  print('-'*80)
  debug = 0
  
  #default_path ="/Users/saguinag/PythonProjects/Phoenix/PhoenixPython";
  default_path ="/home/saguinag/Phoenix/PhoenixPython";
  chdir(default_path)

  ## Given a graph in gml format
  #  input_graph = "demo_graphs/netscience.gml"
  #  input_graph = "demo_graphs/politicalbooks.gml"
  #  input_graph = "demo_graphs/caveman_3x4.gml"
  input_graph = "karate_Zachary.gml"
  #  input_graph = "demo_graphs/toy_graph.gml"

  ## graph.cliques
  g         = load_graph(input_graph)
  
  g_cliques = list(nx.find_cliques(g))
  
  ## graph.compress
  print '-'*80
  print ('Starting compression ...')
  
  ## clique distribution
  clq_dist = helpers.clique_number_distribution(g_cliques)
  
  ## clique to clique intersection distribution
  clq2clq_intxn_dist = phoenix.clique2cliqueIntersection(g_cliques)
  
  
  clq_dist = helpers.normalize_distribution(clq_dist)
  clq2clq_intxn_dist = helpers.normalize_distributions(clq2clq_intxn_dist)
  
  if debug: print " ", clq_dist.keys()
  if debug: print " ", clq2clq_intxn_dist.keys()
  if debug: print '  Graph compressed. Distributions normalized.'

  print '.'*80,'\nGraph Formation...'
  leGraph = graph_formation(clq_dist, clq2clq_intxn_dist, .5, 36, True)
  print ' ',nx.diameter(g), nx.diameter(leGraph), leGraph.number_of_nodes()
  ##
#  g_clqs = list(nx.find_cliques(leGraph))
#  c_dist = helpers.clique_number_distribution(g_clqs)
#  c_dist = helpers.normalize_distribution(clq_dist)
#  if debug: print ' ', divtools.jsd2(clq_dist, c_dist)
  #pp(clq_dist)
  #pp(c_dist)
  fig, ax = plt.subplots(1, 2, figsize=(1.6*9,1*9))
  nx.draw_networkx(g, ax=ax[0])
  nx.draw_networkx(leGraph, ax=ax[1])
  plt.show()
  #print len(g.nodes())





# http://www.cs.toronto.edu/~urtasun/courses/GraphicalModels/graphical_models.html
# http://stackoverflow.com/questions/25222322/networkx-create-new-graph-of-all-nodes-that-are-a-part-of-4-node-clique
# https://gist.github.com/conradlee/1341985
