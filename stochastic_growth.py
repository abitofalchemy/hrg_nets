__author__ = 'tweninge'

import random
import math
import networkx as nx
import net_metrics as metrics

def weighted_choice(choices):
    total = sum(w for c, w in choices)
    r = random.uniform(0, total)
    upto = 0
    for c, w in choices:
        if upto + w > r:
            return c
        upto += w
    assert False, "Shouldn't get here"


F = .6  # math.e


def control_rod(choices, H, num_nodes):
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
        wt = t#.1  # t*-wn

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


def try_combination(lhs, N):
    lhs
    for i in range(0, len(N)):
        n = N[i]
        if lhs[0] == "S":
            break
        if len(lhs) == len(n):
            t = []
            for i in n:
                t.append(i)
            random.shuffle(t)
            return zip(t, lhs)
    return []


def find_match(N, prod_rules):
    if len(N) == 1 and ['S'] in N: return [('S', 'S')]
    matching = {}
    while True:
        lhs = random.choice(prod_rules.keys()).lstrip("(").rstrip(")")  ##TODO make this weighted choice
        lhs = lhs.split(",")
        c = try_combination(lhs, N)
        if c: return c

"""
import graph_sampler as gs
def triangle_match_test(lhs_match, H, he):
    match = []
    e = []
    for tup in lhs_match:
        match.append(tup[0])
        e.append(tup[1])

    if len(match) == 2 :
        G = construct(H)
        if G.has_edge(match[0], match[1]):
            if '0,a:T' in he and '0,b:T' in he:
                triangles[2]+=1
        else:
            for k,v in nx.single_source_shortest_path(G, match[0], 2).items():
                if len(v)==3 and v[2] == match[1]:
                    triangles[3]+=1
    elif len(lhs_match) == 1:
        if '0,a:T' in he and '1,a:T' in he and '0,1:T' in he:
            triangles[1]+=1


def construct(H):
    G = nx.Graph()
    for ed in H:
        if (len(ed) == 1):
            G.add_node(ed[0])
        else:
            G.add_edge(ed[0], ed[1])
    return G
"""

def grow(prod_rules, n, diam=0):
    D = list()
    newD = diam
    H = list()
    N = list()
    N.append(["S"])  # starting node
    ttt = 0
    # pick non terminal
    num = 0
    while len(N) > 0 and num < n:
        lhs_match = find_match(N, prod_rules)
        e = []
        match = []
        for tup in lhs_match:
            match.append(tup[0])
            e.append(tup[1])
        lhs_str = "(" + ",".join(str(x) for x in sorted(e)) + ")"


        # DO SOMETHING USEFUL WITH THIS MATCH
        new_idx = {}
        n_rhs = str(control_rod( prod_rules[lhs_str].items(), H, n )).lstrip("(").rstrip(")")
        # print lhs_str, "->", n_rhs
        #triangle_match_test(lhs_match, H, n_rhs)
        for x in n_rhs.split(")("):
            new_he = []
            he = x.split(":")[0]
            term_symb = x.split(":")[1]
            for y in he.split(","):
                if y.isdigit():  # y is internal node
                    if y not in new_idx:
                        new_idx[y] = num
                        num += 1
                        #if diam > 0 and num>=newD:
                            #newD=newD+diam
                            #newG = nx.Graph()
                            #for e in H:
                            #    if (len(e) == 1):
                             #       newG.add_node(e[0])
                             #   else:
                            #       newG.add_edge(e[0], e[1])
                            #D.append(metrics.bfs_eff_diam(newG, 20, 0.9))
                            #print "H size:" + str(num)
                    new_he.append(new_idx[y])
                else:  # y is external node
                    for tup in lhs_match:  # which external node?
                        if tup[1] == y:
                            new_he.append(tup[0])
                            break
            # prod = "(" + ",".join(str(x) for x in new_he) + ")"
            if term_symb == "N":
                N.append(sorted(new_he))
            elif term_symb == "T":
                H.append(new_he)
        match = sorted(match)
        N.remove(match)


            #xxx = metrics.bfs_eff_diam(newG, 20, 0.9)
            #if (xxx < ttt):
            #    print str(xxx) + " Shrink"
            #ttt = xxx


    newG = nx.Graph()
    for e in H:
        if (len(e) == 1):
            newG.add_node(e[0])
        else:
            newG.add_edge(e[0], e[1])

    return newG, D

    # print newG.edges()
        # print "V = ", newG.number_of_nodes()
        # print "E = ", newG.number_of_edges()
        # giant_nodes = max(nx.connected_component_subgraphs(newG), key=len)
        # giant = nx.subgraph(newG, giant_nodes)
        # print "V in giant component = ", giant.number_of_nodes()
        # print "E in giant compenent = ", giant.number_of_edges()
        # print "Diameter = ", nx.diameter(nx.subgraph(newG, giant))