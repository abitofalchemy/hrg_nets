import random
from bisect import insort
from collections import defaultdict
import networkx as nx
from pprint import pprint as pp
from .SortedKeyCollections import SortedKeyDefaultDict, SortedKeyCounter
from .helpers import max_intersecting_clique, merge, get_weighted_random_value


def make_graph_from_maximal_cliques(max_cliques, iter_nbr):
    from itertools import combinations
    import matplotlib.pyplot as plt
    import datetime

    if len(max_cliques) is 0:
        print ('Error: clique is empty')
        return
    edg_lst = []
    for clq in max_cliques:
        #print clq
        edg_lst.append(list(combinations(clq, 2)))

    ## Generate a plot
    g = nx.Graph()
    for el in edg_lst:
        g.add_edges_from(el)
    plt.clf()
    nx.draw(g)
    now_str ="{:%d%b_%I%M%S}".format(datetime.datetime.now()) +"_%d".format(iter_nbr) +'.png';
    now_str ="out_%d.png"%iter_nbr
    out_png = "/tmp/" + now_str
    plt.savefig(out_png)
    ##
    return# edg_lst


def make_clique_edges(nodes):
    from itertools import combinations
    return list(combinations(nodes, 2))

def clique2cliqueIntersection(cliques):
  """
  30Sep15.SA: a clone of the orig compress, but only return the 2 to 1 intxn cnt
  """
  one_to_two_model = defaultdict(SortedKeyCounter)
  two_to_one_intxn_model = SortedKeyDefaultDict(SortedKeyCounter)
  cliques = sorted(cliques, key=lambda x: len(x))
  #print (cliques)
  
  while len(cliques) > 1:
      collapsing_cliq = cliques.pop()
      
      max_intxn_clq = max_intersecting_clique(collapsing_cliq, cliques)
      
      if max_intxn_clq:
          cliques.remove(max_intxn_clq)
          
          
          intxn = set(collapsing_cliq).intersection(max_intxn_clq)
          if (not len(intxn)):
            print ('Cliques left:', len(cliques))
            print ('intersection is of length 0!!!!! ')
              
            break
        
          # Update inner intersection models:
          two_to_one_intxn_model[(len(collapsing_cliq), len(max_intxn_clq))][len(intxn) or 1] += 1
          
#          merged_clq = merge(collapsing_cliq, max_intxn_clq)

#          merged_clq = set(merged_clq).difference(intxn)

          # Here's where we encode the compression
          #one_to_two_model[len(merged_clq)][(len(collapsing_cliq), len(max_intxn_clq))] += 1
  
  return two_to_one_intxn_model  # , out_clq_intxn_model

def compress(cliques):
    """"""
    one_to_two_model = defaultdict(SortedKeyCounter)

    two_to_one_intxn_model = SortedKeyDefaultDict(SortedKeyCounter)

    cliques = sorted(cliques, key=lambda x: len(x))
    #print (cliques)

    while len(cliques) > 1:
        collapsing_cliq = cliques.pop()

        max_intxn_clq = max_intersecting_clique(collapsing_cliq, cliques)

        if max_intxn_clq:
            cliques.remove(max_intxn_clq)


            intxn = set(collapsing_cliq).intersection(max_intxn_clq)
            if (not len(intxn)):
                print ('Cliques left:', len(cliques))
                print ('intersection is of length 0!!!!! ')

                break

            # Update inner intersection models:
            two_to_one_intxn_model[(len(collapsing_cliq), len(max_intxn_clq))][len(intxn) or 1] += 1

            merged_clq = merge(collapsing_cliq, max_intxn_clq)

            merged_clq = set(merged_clq).difference(intxn)

            # Here's where we encode the compression
            one_to_two_model[len(merged_clq)][(len(collapsing_cliq), len(max_intxn_clq))] += 1

    return one_to_two_model, two_to_one_intxn_model  # , out_clq_intxn_model


sorted_max_intxng_clique_compression = compress


def generate_hypergraph(seed_graph, one_to_two_model, two_to_one_intxn_model,
                        clique_substitution_model, outer_intxn_size_model,
                        node_limit, debug=False):
    """ 
    where argument 3 is clique_substitution_model = clq_numb_dist
    and outer_intxn_size_model = clqs_x_clqs_dist

    Issues:
    What is node_limit?
    how it works:
    - given a seed, find its cliques, count the nodes
    - while we grow the node count we swap-out cliques as a way to reconstruct 
      the original graph
    """
    clqs = list(nx.find_cliques(seed_graph))
    # clqs = seeds
    clq_nums = [len(clq) for clq in clqs]
    nodes = list(seed_graph.nodes())
    node_count = len(nodes)

    ## debug
    k = 0
    while node_count < node_limit:

        node_count = len(set(nx.utils.flatten(clqs)))
        debug = False
        #print('-'*40, ' ',k, 'iter,', 'node_cout', node_count)
        k += 1

        if node_count >= node_limit:
            print('|V|:', node_count)
            break

        # if debug: print("###############################")
        # if debug: print("###### ITERATION NO. {0} ######".format(i+1))

        clq_size_to_swap_out = get_weighted_random_value(clique_substitution_model)
        if not debug:
            print("clq_size_to_swap_out", clq_size_to_swap_out)
            #print("clq_size_to_swap_out, clq.len:", clq_size_to_swap_out, len(clq))

        possible_swap_out_clqs = [clq for clq in clqs if len(clq) == clq_size_to_swap_out]

        if not possible_swap_out_clqs:
            #print 'continuing ...'
            continue
        else:
            ## Return a k length list of unique elements chosen from the pop. seq. rand sampling w/o replacement.
            swap_out_clq = random.sample(possible_swap_out_clqs, 1)[0]
        if debug: print("swap_out_clq:", swap_out_clq)

        clqs.remove(swap_out_clq)

        try:
            #pp( one_to_two_model)
            # pp( one_to_two_model[clq_size_to_swap_out])
            new_clqs_sizes = get_weighted_random_value(one_to_two_model[clq_size_to_swap_out])
        except (IndexError, KeyError):
            # %%debug
            #print("IndexError:", one_to_two_model, clq_size_to_swap_out)
            continue
        if debug: print("  new_clqs_sizes:", new_clqs_sizes)

        inner_intxn_size = get_weighted_random_value(two_to_one_intxn_model[new_clqs_sizes])
        if debug: print("inner_intxn_size:", inner_intxn_size)

        # We can calculate the number of new nodes in our graph:
        new_nodes_count = sum(new_clqs_sizes) - inner_intxn_size - len(swap_out_clq)
        if debug: print("number_of_new_nodes", new_nodes_count)

        new_nodes = list(range(node_count, node_count + new_nodes_count))
        # new_nodes = list(range(node_count, node_count+inner_intxn_size))
        intxn_nodes, new_nodes = new_nodes[:inner_intxn_size], new_nodes[inner_intxn_size:]
        if debug: print("intxn_nodes, new_nodes:", intxn_nodes, new_nodes)

        # We initialize the two new cliques with the intersection nodes
        first_clq = list(intxn_nodes)
        second_clq = list(intxn_nodes)
        # print(first_clq, second_clq)

        filler_nodes = swap_out_clq + new_nodes  ## combining two lists
        if debug: print("filler_nodes:", filler_nodes)

        for enum, node in enumerate(filler_nodes):
            if enum % 2 == 0:
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
        #make_graph_from_maximal_cliques(clqs, k)

        # print('1', outer_intxn_size_model[len(first_clq)])  #['intersects']
        # print('2', outer_intxn_size_model[len(second_clq)]) #['intersects']
        # try:
        #     first_clq_branch = get_weighted_random_value(outer_intxn_size_model[len(first_clq)]['intersects'])
        #     #first_clq_branch = get_weighted_random_value(outer_intxn_size_model[len(first_clq)])
        # except (KeyError):
        #     first_clq_branch = False
        #     pass
        # try:
        #     second_clq_branch = get_weighted_random_value(outer_intxn_size_model[len(second_clq)]['intersects'])
        #     #second_clq_branch = get_weighted_random_value(outer_intxn_size_model[len(second_clq)])
        # except (KeyError):
        #     second_clq_branch = False
        #     pass
        #
        # if not debug: print("first_clq_branch:", first_clq_branch)
        # if not debug: print("second_clq_branch:", second_clq_branch)
        #
        #
        # if first_clq_branch:
        #     print("Hit: first")
        #     random_clq = random.sample(clqs, 1)[0]
        #
        #     if random_clq != first_clq:
        #
        #         if debug: print("random_clq: {0} - connecting_clq: {1}".format(random_clq, first_clq))
        #
        #         try:
        #             clique_intxn_dist = outer_intxn_size_model[len(first_clq)][len(random_clq)]
        #         except (KeyError):
        #             continue
        #
        #         clique_intxn_dist = {k: v for k, v in clique_intxn_dist.items() if k != 0}
        #
        #         normalizing_value = sum(clique_intxn_dist.values())
        #
        #         clique_intxn_dist = {k: (v / normalizing_value) for k, v in clique_intxn_dist.items()}
        #
        #         if debug: print("clique_intxn_dist:", clique_intxn_dist)
        #
        #         if clique_intxn_dist:
        #             intxn_size = get_weighted_random_value(clique_intxn_dist)
        #
        #             if debug: print("intxn_size:", intxn_size)
        #
        #             cnnct_from_itxn_nodes = random.sample(first_clq, intxn_size)
        #             if debug: print("cnnct_from_itxn_nodes:", cnnct_from_itxn_nodes)
        #
        #             cnnct_to_intxn_nodes = random.sample(random_clq, 1)
        #             if debug: print("cnnct_to_intxn_nodes:", cnnct_to_intxn_nodes)
        #
        #             new_branch = list(cnnct_from_itxn_nodes) + list(cnnct_to_intxn_nodes)
        #
        #             new_branch = list(set(new_branch))
        #
        #             if debug: print("new_branch:", new_branch)
        #
        #             clqs.append(new_branch)
        # if second_clq_branch:
        #     print("Hit: second")
        #     random_clq = random.sample(clqs, 1)[0]
        #
        #     if random_clq != second_clq:
        #
        #         if not debug: print("random_clq: {0} - connecting_clq: {1}".format(random_clq, second_clq))
        #
        #         try:
        #             clique_intxn_dist = outer_intxn_size_model[len(second_clq)][len(random_clq)]
        #         except (KeyError):
        #             continue
        #
        #         clique_intxn_dist = {k: v for k, v in clique_intxn_dist.items() if k != 0}
        #
        #         normalizing_value = sum(clique_intxn_dist.values())
        #
        #         clique_intxn_dist = {k: (v / normalizing_value) for k, v in clique_intxn_dist.items()}
        #
        #         if debug: print("clique_intxn_dist:", clique_intxn_dist)
        #
        #         if clique_intxn_dist:
        #             intxn_size = get_weighted_random_value(clique_intxn_dist)
        #
        #             if debug: print("intxn_size:", intxn_size)
        #
        #             cnnct_from_itxn_nodes = random.sample(second_clq, intxn_size)
        #             if debug: print("cnnct_from_itxn_nodes:", cnnct_from_itxn_nodes)
        #
        #             cnnct_to_intxn_nodes = random.sample(random_clq, 1)
        #             if debug: print("cnnct_to_intxn_nodes:", cnnct_to_intxn_nodes)
        #
        #             new_branch = list(cnnct_from_itxn_nodes) + list(cnnct_to_intxn_nodes)
        #             new_branch = list(set(new_branch))
        #
        #             if debug: print("new_branch:", new_branch)
        #
        #             clqs.append(new_branch)
                    # print(len(clqs))
                    # Keep our total node count updated:
                    # node_count += new_nodes_count
                    # if not debug: print("node_count", node_count)
                    # print('clqs size', len(clqs))
    #print('add eges')
    hypergraph = nx.Graph()
    for clq in clqs:
        hypergraph.add_edges_from(make_clique_edges(clq))
    return hypergraph, clqs
