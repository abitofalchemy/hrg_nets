
# coding: utf-8

# In[1]:

get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import networkx as nx
import collections
from pprint import PrettyPrinter as pp
pp = pp(2).pprint

powergrid_graph = nx.read_gml('c:/Users/rodrigo/github/hyphy/datasets/power.gml')
'''
karate_graph = nx.read_gml('c:/Users/rodrigo/github/hyphy/datasets/karate.gml')

les_mis_graph = nx.read_gml('c:/Users/rodrigo/github/hyphy/datasets/lesmis.gml')

dolphin_graph = nx.read_gml('c:/Users/rodrigo/github/hyphy/datasets/dolphins.gml')

football_graph = nx.read_gml('c:/Users/rodrigo/github/hyphy/datasets/football.gml')
'''
#nx.draw_networkx(powergrid_graph, font_size=10, node_size=200) #,pos=nx.spring_layout(karate_graph))


# In[3]:

from statscounter import StatsCounter
from collections import defaultdict

class SortedKeyStatsCounter(StatsCounter):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        StatsCounter.__init__(self, *args, **kwargs)
    
    def __getitem__(self, key):
        return super(SortedKeyStatsCounter, self).__getitem__(self.__keytransform__(key))

    def __setitem__(self, key, value):
        super(SortedKeyStatsCounter, self).__setitem__(self.__keytransform__(key), value)

    def __delitem__(self, key):
        super(SortedKeyStatsCounter, self).__delitem__(self.__keytransform__(key))

    def __keytransform__(self, key):
        try:
            return tuple(sorted(key))
        except (TypeError):
            return key

        
class SortedKeyDefaultDict(defaultdict):
    """A dictionary that applies an arbitrary key-altering
       function before accessing the keys"""

    def __init__(self, *args, **kwargs):
        defaultdict.__init__(self, *args, **kwargs)
    
    def __getitem__(self, key):
        return super(SortedKeyDefaultDict, self).__getitem__(self.__keytransform__(key))

    def __setitem__(self, key, value):
        super(SortedKeyDefaultDict, self).__setitem__(self.__keytransform__(key), value)

    def __delitem__(self, key):
        super(SortedKeyDefaultDict, self).__delitem__(self.__keytransform__(key))

    def __keytransform__(self, key):
        try:
            return tuple(sorted(key))
        except (TypeError):
            return key    
        
        
sc = SortedKeyStatsCounter()


# In[10]:

def intersection_size(clique_one, clique_two):
    return len(set(clique_one).intersection(clique_two))

def max_intersecting_clique(clq, clqs):
    """
    return the clique (from *clqs*) that produces
    the largest intersection with *clq*
    """
    max_intxn = max(clqs, key=lambda x: intersection_size(x,clq))
    return max_intxn

def merge(clq_a, clq_b):
    return set(clq_a).union(clq_b)


# In[5]:

def get_weighted_random_value(model):
    """
    This will generate a value given a probability distribution
    """
    from bisect import bisect
    from random import random
    #http://stackoverflow.com/questions/4437250/choose-list-variable-given-probability-of-each-variable
    P = list(model.items())
    
    cdf = [P[0][1]]
    for i in range(1, len(P)):
        cdf.append(cdf[-1] + P[i][1])

    return P[bisect(cdf, random())][0]

def intersection_size(clique_one, clique_two):
    return len(set(clique_one).intersection(clique_two))

def max_intersecting_clique(clq, clqs):
    """
    return the clique (from *clqs*) that produces
    the largest intersection with *clq*
    """
    max_intxn = max(clqs, key=lambda x: intersection_size(x,clq))
    return max_intxn

def max_intersection_clique(clq, clqs):
    """
    return the clique (from *clqs*) that produces
    the largest intersection with *clq*
    """
    max_intxn = None
    intxn = None
    max_intxn_size = 0
    
    if len(clqs) == 1:
        return clqs[0], {}
    
    for clique in clqs:
        #print("max_intersection_clique:", intxn)
        #intxn_size = len(set(clq).intersection(clique))
        intxn_size = intersection_size(clq, clique)
        
        if intxn_size > max_intxn_size:
            intxn = set(clq).intersection(clique)
            max_intxn = clique
            max_intxn_size = intxn_size
    
    return max_intxn, intxn


def intersections_with_hypergraph(central_clique, cliques):
    intxn_counter = collections.defaultdict(collections.Counter)
    
    for clique in cliques:
        intxn_counter[(len(central_clique), len(clique))][intersection_size(central_clique, clique)] += 1
    
    return intxn_counter

def merge(clq_a, clq_b):
    return set(clq_a).union(clq_b)


def make_clique_edges(nodes):
    from itertools import combinations
    return list(combinations(nodes, 2))


def clique_number_distribution(cliques):
    clique_number_counts = SortedKeyStatsCounter()

    for clq in cliques:
        clique_number_counts[len(clq)] += 1
        
    return clique_number_counts


def cliques_x_cliques_distribution(cliques, debug=False):
    """"""
    intxn_counters = SortedKeyDefaultDict(SortedKeyStatsCounter)
    
    from itertools import combinations
    
    for enum, (clqA, clqB) in enumerate(combinations(cliques, 2)):
        intxn_size = intersection_size(clqA, clqB)
        if debug: print(enum, clqA, clqB, intxn_size)
        # conditional distributions(?)
        if debug: print(intxn_counters)
        intxn_counters[(len(clqA),len(clqB))][intxn_size] += 1
          
    return intxn_counters


def normalize_distribution(counter):
    """
    Given a *counter* (frequency distribution,
    AKA collections.Counter), normalize the frequencies
    to produce a probability distribution. 
    """
    total = sum(counter.values())
    return counter.__class__({k:v/total for k, v in counter.items()})

def normalize_distributions(distributions):
    return {key:normalize_distribution(val) 
            for key, val in distributions.items()}

def normalize_distribution_of_distributions(dist_of_dists):
    return {key: normalize_distributions(val) 
            for key, val in dist_of_dists.items()}


# In[7]:

def sorted_max_intxng_dblclq_to_clq_compression(cliques):
    from bisect import insort
    
    from collections import defaultdict, Counter
    compression_model = defaultdict(SortedKeyStatsCounter)
    
    in_clq_intxn_model = SortedKeyDefaultDict(SortedKeyStatsCounter)
    
    out_clq_intxn_model = SortedKeyDefaultDict(SortedKeyStatsCounter)
    
    out_temporal_clq_intxn_model = defaultdict(SortedKeyStatsCounter)
    
    cliques = sorted(cliques, key=lambda x: len(x))
    #collapsing_cliq = cliques.pop()
    
    while len(cliques) > 1:
        collapsing_cliq = cliques.pop()

        max_intxn_clq = max_intersecting_clique(collapsing_cliq, cliques)
        
        if max_intxn_clq:
            cliques.remove(max_intxn_clq)
            
            intxn = set(collapsing_cliq).intersection(max_intxn_clq)
            
            # Update inner intersection models:
            in_clq_intxn_model[(len(collapsing_cliq), len(max_intxn_clq))][len(intxn) or 1] += 1
              
            merged_clq = merge(collapsing_cliq, max_intxn_clq)
            #print("pre-diff merged_clq", merged_clq)
            merged_clq = set(merged_clq).difference(intxn)
            #print("post-diff collapsing_cliq", merged_clq)
                    
            # Here's where we encode the compression
            compression_model[len(merged_clq)][(len(collapsing_cliq), len(max_intxn_clq))] += 1
            
    return collapsing_cliq, compression_model, in_clq_intxn_model, out_clq_intxn_model

sorted_max_intxng_clique_compression = sorted_max_intxng_dblclq_to_clq_compression


# In[9]:

def generate_hypergraph(seed_graph, iterations, clique_substitution_model, 
                        production_rules_model, inner_intxn_size_model, 
                        outer_intxn_size_model, intxn_occurance_model,
                        node_limit=999, debug=False):
   
    import random
    hypergraph = nx.Graph()
    clqs = list(nx.find_cliques(seed_graph))
    clq_nums = [len(clq) for clq in clqs]
    nodes = list(seed_graph.nodes())
    node_count = len(nodes)

    #for i in range(iterations):
    while node_count < node_limit:
        
        node_count = len(set(nx.utils.flatten(clqs)))
        
        if node_count >= node_limit:
            print(node_count)
            break
        
        if debug: print("###############################")
        if debug: print("###### ITERATION NO. {0} ######".format(i+1))
        
        clq_size_to_swap_out = get_weighted_random_value(clique_substitution_model)
        if debug: print("clq_size_to_swap_out:", clq_size_to_swap_out)
        
        possible_swap_out_clqs = [clq for clq in clqs if len(clq) == clq_size_to_swap_out]
        
        if not possible_swap_out_clqs:
            continue
        else:           
            swap_out_clq = random.sample(possible_swap_out_clqs, 1)[0]
        if debug: print("swap_out_clq:", swap_out_clq)
        
        clqs.remove(swap_out_clq)
       
        try:
            new_clqs_sizes = get_weighted_random_value(production_rules_model[clq_size_to_swap_out])
        except (IndexError, KeyError):
            #%%debug
            #print("IndexError:", production_rules_model, clq_size_to_swap_out)
            continue
        if debug: print("new_clqs_sizes:", new_clqs_sizes)
        

        inner_intxn_size = get_weighted_random_value(inner_intxn_size_model[new_clqs_sizes])
        if debug: print("inner_intxn_size:", inner_intxn_size)
        
        # We can calculate the number of new nodes in our graph:
        new_nodes_count = sum(new_clqs_sizes) - inner_intxn_size - len(swap_out_clq)
        if debug: print("number_of_new_nodes", new_nodes_count)
        
        new_nodes = list(range(node_count, node_count+new_nodes_count))
        #new_nodes = list(range(node_count, node_count+inner_intxn_size))
        intxn_nodes, new_nodes = new_nodes[:inner_intxn_size], new_nodes[inner_intxn_size:]
        if debug: print("intxn_nodes, new_nodes:", intxn_nodes, new_nodes)
        
        # We initialize the two new cliques with the intersection nodes
        first_clq = list(intxn_nodes)
        second_clq = list(intxn_nodes)
        
        filler_nodes = swap_out_clq + new_nodes
        if debug: print("filler_nodes:", filler_nodes)
        for enum, node in enumerate(filler_nodes):
            if enum % 2 == 0 :
                if len(first_clq) < new_clqs_sizes[0]:
                    first_clq.append(node)
                else:
                    assert len(second_clq) < new_clqs_sizes[1]
                    second_clq.append(node)
            else: 
                if len(second_clq) < new_clqs_sizes[1]:
                    second_clq.append(node)
                else:
                    assert len(first_clq) < new_clqs_sizes[0]
                    first_clq.append(node)
        if debug: print("first_clq, second_clq:", first_clq, second_clq)
        clqs.append(first_clq)
        clqs.append(second_clq)
        if debug: print("clqs:", clqs)
        
        try:
            first_clq_branch = get_weighted_random_value(outer_intxn_size_model[len(first_clq)]['intersects'])
        except (KeyError):
            
            first_clq_branch = False
            pass
        try:
            second_clq_branch = get_weighted_random_value(outer_intxn_size_model[len(second_clq)]['intersects'])
        except (KeyError):
            
            second_clq_branch = False
            pass
        
        if debug: print("first_clq_branch:", first_clq_branch)
        if debug: print("second_clq_branch:", second_clq_branch)
        
        if first_clq_branch:
            random_clq = random.sample(clqs,1)[0]
            
            if random_clq != first_clq:
                
                if debug: print("random_clq: {0} - connecting_clq: {1}".format(random_clq, first_clq))
                    
                try:
                    clique_intxn_dist = outer_intxn_size_model[len(first_clq)][len(random_clq)]
                except (KeyError):
                    continue
                    
                clique_intxn_dist = {k:v for k, v in clique_intxn_dist.items() if k!=0}

                normalizing_value = sum(clique_intxn_dist.values())

                clique_intxn_dist = {k:(v/normalizing_value) for k, v in clique_intxn_dist.items()}

                if debug: print("clique_intxn_dist:", clique_intxn_dist)
                    
                if clique_intxn_dist:
                    intxn_size = get_weighted_random_value(clique_intxn_dist)

                    if debug: print("intxn_size:", intxn_size)

                    cnnct_from_itxn_nodes = random.sample(first_clq, intxn_size)
                    if debug: print("cnnct_from_itxn_nodes:",cnnct_from_itxn_nodes)

                    cnnct_to_intxn_nodes = random.sample(random_clq, 1)
                    if debug: print("cnnct_to_intxn_nodes:",cnnct_to_intxn_nodes)

                    new_branch = list(cnnct_from_itxn_nodes)+list(cnnct_to_intxn_nodes)
                    
                    new_branch = list(set(new_branch))
                    
                    if debug: print("new_branch:",new_branch)
                    
                    clqs.append(new_branch)
        if second_clq_branch:
            random_clq = random.sample(clqs,1)[0]
            
            if random_clq != second_clq:

                if debug: print("random_clq: {0} - connecting_clq: {1}".format(random_clq, second_clq))
                    
                try:
                    clique_intxn_dist = outer_intxn_size_model[len(second_clq)][len(random_clq)]
                except (KeyError):
                    continue
                    
                clique_intxn_dist = {k:v for k, v in clique_intxn_dist.items() if k!=0}

                normalizing_value = sum(clique_intxn_dist.values())

                clique_intxn_dist = {k:(v/normalizing_value) for k, v in clique_intxn_dist.items()}

                if debug: print("clique_intxn_dist:", clique_intxn_dist)

                if clique_intxn_dist:
                    intxn_size = get_weighted_random_value(clique_intxn_dist)

                    if debug: print("intxn_size:", intxn_size)

                    cnnct_from_itxn_nodes = random.sample(second_clq, intxn_size)
                    if debug: print("cnnct_from_itxn_nodes:",cnnct_from_itxn_nodes)

                    cnnct_to_intxn_nodes = random.sample(random_clq, 1)
                    if debug: print("cnnct_to_intxn_nodes:",cnnct_to_intxn_nodes)

                    new_branch = list(cnnct_from_itxn_nodes)+list(cnnct_to_intxn_nodes)
                    new_branch = list(set(new_branch))
                    
                    if debug: print("new_branch:",new_branch)
                    
                    clqs.append(new_branch)
                    
        # Keep our total node count updated:
        #node_count += new_nodes_count
        if debug: print("node_count", node_count)
        
    for clq in clqs:
        hypergraph.add_edges_from(make_clique_edges(clq))
    return hypergraph, clqs


# In[47]:

karate_cliques = list(nx.find_cliques(karate_graph))
karate_cliques = sorted(karate_cliques, key=lambda x: len(x))
les_mis_cliques = list(nx.find_cliques(les_mis_graph))
les_mis_cliques = sorted(les_mis_cliques, key=lambda x: len(x))
dolphin_cliques = list(nx.find_cliques(dolphin_graph))
dolphin_cliques = sorted(dolphin_cliques, key=lambda x: len(x))
#pp(powergrid_cliques)


# In[2]:

powergrid_cliques = list(nx.find_cliques(powergrid_graph))
powergrid_cliques = sorted(powergrid_cliques, key=lambda x: len(x))
#pp(powergrid_cliques)


# In[ ]:

powergrid_clique_numb_dist = clique_number_distribution(powergrid_cliques)
#pp(clique_numb_dist)
powergrid_clqsxclqs_dist = cliques_x_cliques_distribution(powergrid_cliques)
#pp(clqsxclqs_dist)


# In[48]:

karate_clique_numb_dist = clique_number_distribution(karate_cliques)

karate_clqsxclqs_dist = cliques_x_cliques_distribution(karate_cliques)

les_mis_clique_numb_dist = clique_number_distribution(les_mis_cliques)

les_mis_clqsxclqs_dist = cliques_x_cliques_distribution(les_mis_cliques)

dolphin_clique_numb_dist = clique_number_distribution(dolphin_cliques)

dolphin_clqsxclqs_dist = cliques_x_cliques_distribution(dolphin_cliques)


# In[11]:

_, powergrid_compression_model, powergrid_in_intxn_size_model, powergrid_out_intxn_size_model = sorted_max_intxng_clique_compression(powergrid_cliques)


# In[50]:

_, karate_compression_model, karate_in_intxn_size_model, karate_out_intxn_size_model = sorted_max_intxng_clique_compression(karate_cliques)

_, les_mis_compression_model, les_mis_in_intxn_size_model, les_mis_out_intxn_size_model = sorted_max_intxng_clique_compression(les_mis_cliques)

_, dolphin_compression_model, dolphin_in_intxn_size_model, dolphin_out_intxn_size_model = sorted_max_intxng_clique_compression(dolphin_cliques)


# In[12]:

powergrid_clq_num_dist = normalize_distribution(clique_number_distribution(powergrid_cliques))
powergrid_clq_intxn_distribution = normalize_distributions(cliques_x_cliques_distribution(powergrid_cliques))


# In[52]:

karate_clq_num_dist = normalize_distribution(clique_number_distribution(karate_cliques))
karate_clq_intxn_distribution = normalize_distributions(cliques_x_cliques_distribution(karate_cliques))
les_mis_clq_num_dist = normalize_distribution(clique_number_distribution(les_mis_cliques))
les_mis_clq_intxn_distribution = normalize_distributions(cliques_x_cliques_distribution(les_mis_cliques))
dolphin_clq_num_dist = normalize_distribution(clique_number_distribution(dolphin_cliques))
dolphin_clq_intxn_distribution = normalize_distributions(cliques_x_cliques_distribution(dolphin_cliques))


# In[13]:

_, powergrid_compression_model_old, powergrid_in_intxn_size_model_old, powergrid_out_intxn_size_model = compress_hypergraph_no_intxn_merge(powergrid_cliques)


# In[14]:

powergrid_compression_model = normalize_distributions(powergrid_compression_model)
powergrid_in_intxn_size_model = normalize_distributions(powergrid_in_intxn_size_model)


# In[53]:

karate_compression_model = normalize_distributions(karate_compression_model)
karate_in_intxn_size_model = normalize_distributions(karate_in_intxn_size_model)

les_mis_compression_model = normalize_distributions(les_mis_compression_model)
les_mis_in_intxn_size_model = normalize_distributions(les_mis_in_intxn_size_model)

dolphin_compression_model = normalize_distributions(dolphin_compression_model)
dolphin_in_intxn_size_model = normalize_distributions(dolphin_in_intxn_size_model)


# In[15]:

powergrid_hgraph = nx.Graph()
powergrid_hgraph.add_edge(0,1)
powergrid_hgraph, powergrid_hedges = generate_hypergraph(powergrid_hgraph, 1000, powergrid_clq_num_dist, powergrid_compression_model, 
                                      powergrid_in_intxn_size_model, powergrid_clq_intxn_distribution, None, node_limit=4000)


print(len(list(powergrid_hgraph.nodes())))
print(len(list(powergrid_graph.nodes())))


# In[16]:

nx.write_edgelist(powergrid_hgraph, 'c:/Users/rodrigo/github/hyphy/datasets/powerGenerated.txt')


# In[ ]:

powergrid_hgraph = nx.Graph()
powergrid_hgraph.add_edge(0,1)
powergrid_hgraph, powergrid_hedges = generate_hypergraph(powergrid_hgraph, 1000, powergrid_clq_num_dist, powergrid_compression_model, 
                                      powergrid_in_intxn_size_model, powergrid_clq_intxn_distribution, None, node_limit=4000)


print(len(list(powergrid_hgraph.nodes())))
print(len(list(powergrid_graph.nodes())))


# In[49]:

draw_multiple_dual_degree_rank_plot([{'G1':powergrid_graph, 'G2':powergrid_hgraph, 
                     'L1':"powergrid", 'L2':"generated"},
                    {'G1':powergrid_graph, 'G2':powergrid_hgraph, 
                     'L1':"powergrid", 'L2':"generated"}])


# In[30]:

draw_multiple_dual_degree_rank_plot([{'G1':karate_graph, 'G2':karate_hgraph, 
                     'L1':"powergrid", 'L2':"generated"},
                    {'G1':karate_graph, 'G2':karate_hgraph, 
                     'L1':"powergrid", 'L2':"generated"}])


# In[50]:

draw_multiple_dual_hop_plot([{'G1':powergrid_graph, 'G2':powergrid_hgraph, 
                     'L1':"powergrid", 'L2':"generated"},
                    {'G1':powergrid_graph, 'G2':powergrid_hgraph, 
                     'L1':"powergrid", 'L2':"generated"}])


# In[42]:

draw_multiple_dual_hop_plot([{'G1':karate_graph, 'G2':karate_hgraph, 
                     'L1':"powergrid", 'L2':"generated"},
                    {'G1':karate_graph, 'G2':karate_hgraph, 
                     'L1':"powergrid", 'L2':"generated"}])


# In[51]:

draw_multiple_dual_scree_plot([{'G1':powergrid_graph, 'G2':powergrid_hgraph, 
                     'L1':"powergrid", 'L2':"generated"},
                    {'G1':powergrid_graph, 'G2':powergrid_hgraph, 
                     'L1':"powergrid", 'L2':"generated"}])


# In[43]:

draw_multiple_dual_scree_plot([{'G1':karate_graph, 'G2':karate_hgraph, 
                     'L1':"powergrid", 'L2':"generated"},
                    {'G1':karate_graph, 'G2':karate_hgraph, 
                     'L1':"powergrid", 'L2':"generated"}])


# In[29]:

def get_graph_hops(graph):
    from collections import Counter
    from random import sample
    
    c = Counter()
    node = sample(graph.nodes(), 1)[0]
    b = nx.bfs_successors(graph, node)

    for l, h in hops(b, node):
        c[l] += h
        
    return c


def hops(all_succs, start, level=0, debug=False):
    if debug: print("level:", level)
    
    succs = all_succs[start] if start in all_succs else []
    if debug: print("succs:", succs)
    
    lensuccs = len(succs)
    if debug: print("lensuccs:", lensuccs)
    if debug: print()
    if not succs:
        yield level, 0
    else:    
        yield level, lensuccs 
    
    for succ in succs:
        #print("succ:", succ)
        for h in hops(all_succs, succ, level+1):
            yield h

def draw_multiple_dual_degree_rank_plot(graph_pairs_dict):
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    fig1 = plt.figure(figsize=[16,5])
    
    gs = GridSpec(100,100,bottom=0.18,left=0.18,right=0.88)

    ax1 = fig1.add_subplot(gs[:,0:45])
    
    degree_sequence=sorted(nx.degree(graph_pairs_dict[0]['G1']).values(),reverse=True) # degree sequence
    degree_sequence2=sorted(nx.degree(graph_pairs_dict[0]['G2']).values(),reverse=True) # degree sequence
    
    ax1.loglog(degree_sequence,'b-',marker='o')
    ax1.loglog(degree_sequence2,'g',marker='o')
    
    # designate ax2 to span all rows, 
    ax2 = fig1.add_subplot(gs[:,55:99])
    
    degree_sequence=sorted(nx.degree(graph_pairs_dict[1]['G1']).values(),reverse=True) # degree sequence
    degree_sequence2=sorted(nx.degree(graph_pairs_dict[1]['G2']).values(),reverse=True) # degree sequence
    
    ax2.loglog(degree_sequence,'b-',marker='o')
    ax2.loglog(degree_sequence2,'g',marker='o')
    
    ax1.set_title("Degree Rank Distribution")
    ax2.set_title("Degree Rank Distribution")
    ax1.set_ylabel('Degree value')
    ax2.set_ylabel('Degree value')
    ax1.set_xlabel('rank')
    ax2.set_xlabel('rank')

    blue_patch = mpatches.Patch(color='blue', label=graph_pairs_dict[0]['L1'])
    green_patch = mpatches.Patch(color='green', label=graph_pairs_dict[0]['L2'])
    ax1.legend(handles=[blue_patch, green_patch])
    
    blue_patch = mpatches.Patch(color='blue', label=graph_pairs_dict[1]['L1'])
    green_patch = mpatches.Patch(color='green', label=graph_pairs_dict[1]['L2'])
    ax2.legend(handles=[blue_patch, green_patch])
    
    plt.show()
    
    
def draw_multiple_dual_scree_plot(graph_pairs_dict):
    import numpy.linalg
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    
    fig1 = plt.figure(figsize=[16,5])
    
    gs = GridSpec(100,100,bottom=0.18,left=0.18,right=0.88)

    ax1 = fig1.add_subplot(gs[:,0:45])
    
    L1=nx.normalized_laplacian_matrix(graph_pairs_dict[0]['G1'])
    e1 = numpy.linalg.eigvals(L1.A)
    L2=nx.normalized_laplacian_matrix(graph_pairs_dict[0]['G2'])
    e2 = numpy.linalg.eigvals(L2.A)
    
    print(graph_pairs_dict[0]['L1']+": Largest eigenvalue:", max(e1))
    print(graph_pairs_dict[0]['L2']+": Largest eigenvalue:", max(e2))
    #print(label1+": Smallest eigenvalue:", min(e1))
    #print(label2+": Smallest eigenvalue:", min(e2))
    
    ax1.plot(sorted(e1, reverse=True)) # histogram with 100 bins
    ax1.plot(sorted(e2, reverse=True)) # histogram with 100 bins

    ax2 = fig1.add_subplot(gs[:,55:99])
    
    L1=nx.normalized_laplacian_matrix(graph_pairs_dict[1]['G1'])
    e1 = numpy.linalg.eigvals(L1.A)
    L2=nx.normalized_laplacian_matrix(graph_pairs_dict[1]['G2'])
    e2 = numpy.linalg.eigvals(L2.A)
    
    print(graph_pairs_dict[1]['L1']+": Largest eigenvalue:", max(e1))
    print(graph_pairs_dict[1]['L2']+": Largest eigenvalue:", max(e2))
    #print(label1+": Smallest eigenvalue:", min(e1))    
    #print(label2+": Smallest eigenvalue:", min(e2))
    
    ax2.plot(sorted(e1, reverse=True)) # histogram with 100 bins
    ax2.plot(sorted(e2, reverse=True)) # histogram with 100 bins

    #### LEGEND
    blue_patch = mpatches.Patch(color='blue', label=graph_pairs_dict[0]['L1'])
    green_patch = mpatches.Patch(color='green', label=graph_pairs_dict[0]['L2'])
    ax1.legend(handles=[blue_patch, green_patch])
    
    blue_patch = mpatches.Patch(color='blue', label=graph_pairs_dict[1]['L1'])
    green_patch = mpatches.Patch(color='green', label=graph_pairs_dict[1]['L2'])
    ax2.legend(handles=[blue_patch, green_patch])
    
    #### AXES LABELS
    ax1.set_xlabel('rank')
    ax2.set_xlabel('rank')
    ax1.set_ylabel('eigenvalue')
    ax2.set_ylabel('eigenvalue')
    plt.show()
    
    
def draw_multiple_network_value(graph_pairs_dict):
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec

    
    fig1 = plt.figure(figsize=[16,5])
    
    gs = GridSpec(100,100,bottom=0.18,left=0.18,right=0.88)

    ax1 = fig1.add_subplot(gs[:,0:45])
    
    ax1.plot(sorted(nx.eigenvector_centrality(graph_pairs_dict[0]['G1']).values(), reverse=True), 'b-')
    ax1.plot(sorted(nx.eigenvector_centrality(graph_pairs_dict[0]['G2']).values(), reverse=True), 'g')
    
    # designate ax2 to span all rows, 
    ax2 = fig1.add_subplot(gs[:,55:99])
    
    ax2.plot(sorted(nx.eigenvector_centrality(graph_pairs_dict[1]['G1']).values(), reverse=True), 'b-')
    ax2.plot(sorted(nx.eigenvector_centrality(graph_pairs_dict[1]['G2']).values(), reverse=True), 'g')
    
    ax1.set_title("Network Value Distribution")
    ax2.set_title("Network Value Distribution")
    ax1.set_ylabel('network value')
    ax2.set_ylabel('network value')
    ax1.set_xlabel('rank')
    ax2.set_xlabel('rank')

    blue_patch = mpatches.Patch(color='blue', label=graph_pairs_dict[0]['L1'])
    green_patch = mpatches.Patch(color='green', label=graph_pairs_dict[0]['L2'])
    ax1.legend(handles=[blue_patch, green_patch])
    
    blue_patch = mpatches.Patch(color='blue', label=graph_pairs_dict[1]['L1'])
    green_patch = mpatches.Patch(color='green', label=graph_pairs_dict[1]['L2'])
    ax2.legend(handles=[blue_patch, green_patch])
    
    plt.show()


    
def draw_multiple_dual_hop_plot(graph_pairs_dict):
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from collections import Counter
    
    fig1 = plt.figure(figsize=[16,5])
    
    gs = GridSpec(100,100,bottom=0.18,left=0.18,right=0.88)

    ax1 = fig1.add_subplot(gs[:,0:45])
    
    ax1.plot(*zip(*list(get_graph_hops(graph_pairs_dict[0]['G1']).items())), c='#0000FF')
    ax1.plot(*zip(*list(get_graph_hops(graph_pairs_dict[0]['G2']).items())), c='#009900')
    
    # designate ax2 to span all rows, 
    ax2 = fig1.add_subplot(gs[:,55:99])
    
    ax2.plot(*zip(*list(get_graph_hops(graph_pairs_dict[1]['G1']).items())), c='#0000FF')
    ax2.plot(*zip(*list(get_graph_hops(graph_pairs_dict[1]['G2']).items())), c='#009900')
    
    ax1.set_title("Hop plot")
    ax2.set_title("Hop plot")
    ax1.set_ylabel('neighbors')
    ax2.set_ylabel('neighbors')
    ax1.set_xlabel('hops')
    ax2.set_xlabel('hops')

    blue_patch = mpatches.Patch(color='blue', label=graph_pairs_dict[0]['L1'])
    green_patch = mpatches.Patch(color='green', label=graph_pairs_dict[0]['L2'])
    ax1.legend(handles=[blue_patch, green_patch])
    
    blue_patch = mpatches.Patch(color='blue', label=graph_pairs_dict[1]['L1'])
    green_patch = mpatches.Patch(color='green', label=graph_pairs_dict[1]['L2'])
    ax2.legend(handles=[blue_patch, green_patch])

    plt.show()


    
def draw_multiple_transivity_nodecounts(graph_pairs_dict):
    import matplotlib.patches as mpatches
    from matplotlib.gridspec import GridSpec
    from collections import Counter
    
    fig1 = plt.figure(figsize=[16,5])
    
    gs = GridSpec(100,100,bottom=0.18,left=0.18,right=0.88)

    ax1 = fig1.add_subplot(gs[:,0:45])
    
    nodecount_transitivity = [(len(g.nodes()), nx.transitivity(g)) for g in graph_pairs_dict[0]['G1']]
    
    ax1.plot(*zip(*nodecount_transitivity), c='#0000FF')
    
    # designate ax2 to span all rows, 
    ax2 = fig1.add_subplot(gs[:,55:99])
    
    nodecount_transitivity2000 = [(len(g.nodes()), nx.transitivity(g)) for g in graph_pairs_dict[1]['G1']]
    
    ax2.plot(*zip(*nodecount_transitivity2000), c='#0000FF')
    
    ax1.set_title("Transitivity vs. Node count")
    ax2.set_title("Transitivity vs. Node count")
    ax1.set_ylabel('Transitivity')
    ax2.set_ylabel('Transitivity')
    ax1.set_xlabel('Count')
    ax2.set_xlabel('Count')

    blue_patch = mpatches.Patch(color='blue', label=graph_pairs_dict[0]['L1'])
    ax1.legend(handles=[blue_patch])
    
    blue_patch = mpatches.Patch(color='blue', label=graph_pairs_dict[1]['L1'])
    ax2.legend(handles=[blue_patch])

    plt.show()
    
    
def draw_multiple_nodecount_degreecount_plots(graph_pairs_dict):
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches
    
    fig1 = plt.figure(figsize=[16,5])
    
    gs = GridSpec(100,100,bottom=0.18,left=0.18,right=0.88)

    ax1 = fig1.add_subplot(gs[:,0:45])
    
    nodecount_degreecount = [(len(g.nodes()), len(g.edges())) for g in graph_pairs_dict[0]['G1']]
    
    ax1.plot(*zip(*nodecount_degreecount))
    
    # designate ax2 to span all rows, 
    ax2 = fig1.add_subplot(gs[:,55:99])
    
    nodecount_degreecount = [(len(g.nodes()), len(g.edges())) for g in graph_pairs_dict[1]['G1']]
    
    ax2.plot(*zip(*nodecount_degreecount))
    
    ax1.set_title("Degree Count vs. Node count - ("+graph_pairs_dict[0]['L1']+")")
    ax2.set_title("Degree Count vs. Node count - ("+graph_pairs_dict[1]['L1']+")")    
    ax1.set_ylabel('Degree count')
    ax2.set_ylabel('Degree count')
    ax1.set_xlabel('Node count')
    ax2.set_xlabel('Node count')
    
    plt.show()
    
    
def draw_multiple_nodecount_avgdegree_plots(graph_pairs_dict):
    from matplotlib.gridspec import GridSpec
    import matplotlib.patches as mpatches
    
    fig1 = plt.figure(figsize=[16,5])
    
    gs = GridSpec(100,100,bottom=0.18,left=0.18,right=0.88)

    ax1 = fig1.add_subplot(gs[:,0:45])    
    
    nodecount_avgdegree = [(len(g.nodes()), sum(nx.degree(g).values())/len(g.nodes())) for g in graph_pairs_dict[0]['G1']]
    
    ax1.plot(*zip(*nodecount_avgdegree))
    
    # designate ax2 to span all rows, 
    ax2 = fig1.add_subplot(gs[:,55:99])
    
    nodecount_avgdegree = [(len(g.nodes()), sum(nx.degree(g).values())/len(g.nodes())) for g in graph_pairs_dict[1]['G1']]
    
    ax2.plot(*zip(*nodecount_avgdegree))
    
    ax1.set_title("Avg. Degree vs. Node Count - ("+graph_pairs_dict[0]['L1']+")")
    ax2.set_title("Avg. Degree vs. Node Count - ("+graph_pairs_dict[1]['L1']+")")
    ax1.set_ylabel('Avg. Degree')
    ax2.set_ylabel('Avg. Degree')
    ax1.set_xlabel('Node count')
    ax2.set_xlabel('Node count')
    
    plt.show()


# In[56]:

get_ipython().magic('debug')


# In[ ]:



