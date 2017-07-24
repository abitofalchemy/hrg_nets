#!/Users/saguinag/ToolSet/anaconda/bin/python
# -*- coding: utf-8 -*-

__author__ = "Sal Aguinaga"
__copyright__ = "Copyright 2015, The Phoenix Project"
__credits__ = ["Sal Aguinaga", "Rodrigo Palacios", "Tim Weninger"]
__license__ = "GPL"
__version__ = "0.1.1"
__maintainer__ = "Sal Aguinaga"
__email__ = "saguinag (at) nd dot edu"
__status__ = "Development"

""" StarLog
# 0.1.1 initial versioned forked from unit_tst_phoenix.py

"""
import phoenix, helpers
from pprint import PrettyPrinter as pp
pp = pp(2).pprint
import networkx as nx
import datetime, timeit
import pickle
#%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

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
    print(datetime.datetime.now()-t0,' elapsed time.')
    return data

def getCliques(g, filename):
    netscience_graph = g
    t0 = datetime.datetime.now()
    cliques = list(nx.find_cliques(netscience_graph))
    print(datetime.datetime.now()-t0,' elapsed time.')
    pickleFileName = "../demo_graphs/"+filename+".pkl"
    with open(pickleFileName, 'wr') as outf:
        pickle.dump(cliques, outf, protocol=pickle.HIGHEST_PROTOCOL)
    return cliques

def load_graph(filename):
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




if __name__ == '__main__':
    print('-'*80)
    debug = False
    # Given 
    input_graph = "../demo_graphs/netscience.gml"
    input_graph = "../demo_graphs/politicalbooks.gml"
    input_graph = "../demo_graphs/caveman_3x4.gml"
    input_graph = "../demo_graphs/karate_club.gml"




    start_time = datetime.datetime.now()   
    if not cliques_exists_for(input_graph):
        g = load_graph(input_graph)
        cliques = getCliques(g, input_graph)
    else:
        cliques  = load_cliques_obj(input_graph)

    cliques = list(cliques)

