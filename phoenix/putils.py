import pickle,datetime
from itertools import combinations
#from phoenix import helpers

#def add_2Graph_clique_along_intersection(g, k, pairs):
#  for pair in pairs:
#    intxn   = helpers.get_weighted_random_value(intx_cliq_freq[pair])
#    print intxn
#  exit()
#    
##    n_tot = len(g.nodes()) + k - intxn
##    a  = pair[0]
##    #print n_tot, a
##    c1 = range(0,a[0])
##    c2 = range(n_tot - a[1], n_tot)
##    print ' ',c1
##    print ' ',c2
##    g.add_edges_from(list(combinations(c1,2)))
##    g.add_edges_from(list(combinations(c2,2)))
#
#
#  return g




def graph_from_clique(k):
  import networkx as nx
  
  nodes = range(0,k)
  edg_lst = list(combinations(nodes,2))
  g = nx.Graph()
  g.add_edges_from(edg_lst)
  return g

#def anneal_clique_toGraph(graph,clique):
#  ## check that graph is an nx.Graph() object
#  if graph.nodes() is None or clique == 0:
#    print 'Huston'
#    return
##
##  print graph
#
#  if len(graph.nodes()) == 0:
#    nodes = range(0,clique)
#    edg_list = list(combinations(nodes,2))
#    graph.add_edges_from(edg_list)
#    return graph
#  
#  else:
#    print 'graph nodes > 0', graph.nodes()


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
    print(datetime.datetime.now()-t0,' elapsed time.')
    return data

def getCliques(g, filename):
    import networkx as nx
    netscience_graph = g
    t0 = datetime.datetime.now()
    cliques = list(nx.find_cliques(netscience_graph))
    print(datetime.datetime.now()-t0,' elapsed time.')
    pickleFileName = "../demo_graphs/"+filename+".pkl"
    with open(pickleFileName, 'wr') as outf:
        pickle.dump(cliques, outf, protocol=pickle.HIGHEST_PROTOCOL)
    return cliques

def load_graph(filename):
    import networkx as nx
    import datetime
    t0 = datetime.datetime.now()
    #netscience_graph = nx.read_gml('/data/saguinag/datasets/arxiv/cond-mat-2003.gml')
    if (filename is None):
        print('! Error no filename')
        return None
    print('Loading ' + filename)
    graph = nx.read_gml('../demo_graphs/'+filename)
    print(datetime.datetime.now()-t0,' elapsed time.')
    return (graph)

def cliques_exists_for(filename):
    from os import path
    pickleFileName = "../demo_graphs/"+filename+".pkl"
    print('Looking for '+pickleFileName)
    if path.exists(pickleFileName):
        return True
    else :
        return False


def draw_adjacency_matrix(G, node_order=None, partitions=[], colors=[]):
    """
    - G is a networkx graph
    - node_order (optional) is a list of nodes, where each node in G
          appears exactly once
    - partitions is a list of node lists, where each node in G appears
          in exactly one node list
    - colors is a list of strings indicating what color each
          partition should be
    If partitions is specified, the same number of colors needs to be
    specified.
    """
    from matplotlib import pyplot, patches
    import matplotlib
    matplotlib.style.use('ggplot')
    import networkx as nx
    import numpy as np

    adjacency_matrix = nx.to_numpy_matrix(G, dtype=np.bool, nodelist=node_order)

    #Plot adjacency matrix in toned-down black and white
    fig = pyplot.figure(figsize=(5, 5)) # in inches
    pyplot.imshow(adjacency_matrix,
                  cmap="Greys",
                  interpolation="none")

    # The rest is just if you have sorted nodes by a partition and want to
    # highlight the module boundaries
    assert len(partitions) == len(colors)
    ax = pyplot.gca()
    for partition, color in zip(partitions, colors):
        current_idx = 0
        for module in partition:
            ax.add_patch(patches.Rectangle((current_idx, current_idx),
                                          len(module), # Width
                                          len(module), # Height
                                          facecolor="none",
                                          edgecolor=color,
                                          linewidth="1"))
            current_idx += len(module)
