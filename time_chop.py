import networkx as nx
import datetime
from collections import Counter as mset


num_to_word_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
num_to_word_history = {}
def num_to_word(num):
    if num not in num_to_word_history:
        base = len(num_to_word_letters)
        place = 1
        result = []
        while num >= 0:
            result.insert(0, num_to_word_letters[(num % base ** place) // base ** (place - 1)])
            num -= base ** place
            place += 1
        num_to_word_history[num] = ''.join(result)
    return num_to_word_history[num]


def load_timefile(f):
    time_dict = {}
    with open(f, "r") as fi:
        for line in fi:
            line = line.strip()
            if line.startswith("#"): continue
            timeAr = line.split("\t")
            id = timeAr[0]
            time = timeAr[1]
            dt = datetime.datetime.strptime(time, '%Y-%m-%d')
            if dt not in time_dict:
                time_dict[dt] = []
            time_dict[dt].append(id)

    return time_dict


def load_koblenz_quad(f):
    graph = nx.Graph()
    time_edge = {}

    with open(f, "r") as fi:
        for line in fi:
            line = line.strip()
            if line.startswith("%"): continue
            timeAr = line.split()
            u = int(timeAr[0])
            v = int(timeAr[1])

            # these folks meet many times... singularified
            if not graph.has_edge(int(u),int(v)):
                graph.add_edge(int(u),int(v))
                w = int(timeAr[2])
                t = (int(timeAr[3]))/100

                if t not in time_edge:
                    time_edge[t] = []
                time_edge[t].append((u, v))

    return graph, time_edge

def small_example():
    graph = nx.Graph()
    time_edge = {}
    time_edge[1] = [(2, 3), (2,4), (3, 5), (4,5), (3,4)]
    time_edge[2] = [(1, 2), (1,5)]
    #time_edge[3] = [(4,6)]
    #time_edge[4] = []
    for x in time_edge.keys():
        for e in time_edge[x]:
            graph.add_edge(e[0], e[1])
    return graph, time_edge

def board_example():
    graph = nx.Graph()
    time_edge = {}
    time_edge[1] = []
    time_edge[2] = []
    time_edge[3] = []
    time_edge[4] = []
    time_edge[5] = []
    time_edge[6] = []
    time_edge[7] = []
    time_edge[8] = []
    time_edge[9] = []

    graph.add_edge(1, 2)
    time_edge[1].append((1, 2))
    graph.add_edge(2, 3)
    time_edge[2].append((2, 3))
    graph.add_edge(3, 4)
    time_edge[3].append((3, 4))
    graph.add_edge(4, 5)
    time_edge[4].append((4, 5))
    graph.add_edge(5, 6)
    time_edge[5].append((5, 6))
    graph.add_edge(6, 1)
    time_edge[5].append((6, 1))
    graph.add_edge(1, 5)
    time_edge[5].append((1, 5))

    graph.add_edge(1, 3)
    time_edge[7].append((1, 3))
    graph.add_edge(1, 4)
    time_edge[8].append((1, 4))

    return graph, time_edge


def board_example_compress():
    graph = nx.Graph()
    time_edge = {}
    time_edge[1] = []
    time_edge[2] = []
    time_edge[3] = []
    time_edge[4] = []
    time_edge[5] = []
    time_edge[6] = []
    time_edge[7] = []
    time_edge[8] = []
    time_edge[9] = []
    time_edge[10] = []

    graph.add_edge(1, 2)
    time_edge[4].append((1, 2))
    graph.add_edge(1, 3)
    time_edge[1].append((1, 3))
    graph.add_edge(1, 4)
    time_edge[2].append((1, 4))
    graph.add_edge(1, 5)
    time_edge[3].append((1, 5))

    graph.add_edge(2, 6)
    time_edge[5].append((2, 6))
    graph.add_edge(2, 7)
    time_edge[5].append((2, 7))
    graph.add_edge(6, 7)
    time_edge[5].append((6, 7))

    graph.add_edge(3, 8)
    time_edge[6].append((3, 8))
    graph.add_edge(3, 9)
    time_edge[6].append((3, 9))
    graph.add_edge(8, 9)
    time_edge[6].append((8, 9))

    graph.add_edge(4, 10)
    time_edge[7].append((4, 10))
    graph.add_edge(4, 11)
    time_edge[7].append((4, 11))
    graph.add_edge(10, 11)
    time_edge[7].append((10, 11))

    graph.add_edge(5, 12)
    time_edge[7].append((5, 12))
    graph.add_edge(5, 13)
    time_edge[7].append((5, 13))
    graph.add_edge(12, 13)
    time_edge[7].append((12, 13))

    graph.add_edge(9, 11)
    time_edge[8].append((9, 11))

    return graph, time_edge


def insert_rule(lhs, rhs):
    nodes = lhs.split(",")
    outer_verts = {}
    inner_verts = {}
    i = 0
    if 'S' in lhs:
        outer_verts['S'] = 'S'
    else:
        for x in nodes:
            outer_verts[x] = num_to_word(i)
            i += 1

    rhs_list = []
    i = 1
    for x in rhs:
        if isinstance(x, str):
            # hyperedge
            rhs_l = []
            for y in x.split(","):
                if y in outer_verts:
                    y = outer_verts[y]
                elif y in inner_verts:
                    y = inner_verts[y]
                else:
                    inner_verts[y] = str(i)
                    y = str(i)
                    i += 1
                rhs_l.append(y)
            rhs_list.append("("+",".join(sorted(rhs_l))+":N)")
        else:
            rhs_l = []
            for y in x:  # x is a terminal edge (tuple)
                y = str(y)
                if y in outer_verts:
                    y = outer_verts[y]
                elif y in inner_verts:
                    y = inner_verts[y]
                else:
                    inner_verts[y] = str(i)
                    y = str(i)
                    i += 1
                rhs_l.append(y)
            rhs_list.append("("+",".join(sorted(rhs_l))+":T)")
    lhs_str = ",".join(sorted(outer_verts.values()))
    rhs_str = "".join(sorted(rhs_list))
    print lhs_str + " => " + rhs_str

    if lhs_str in production_rules:
        production_rules[lhs_str][rhs_str] += 1
    else:
        rhs_mset = mset()
        rhs_mset[rhs_str] += 1
        production_rules[lhs_str] = rhs_mset


production_rules = {}

G, T = load_koblenz_quad("../demo_graphs/haggle_contact_koblenz.txt")
# G, T = load_koblenz_quad("../demo_graphs/out.topology")
# G, T = board_example()
# G, T = board_example_compress()
# G, T = small_example()

N = []
GT = nx.DiGraph()
R = []
seen = set()
rhs_dict = {}

for key in sorted(T.keys()):
    # backwards version
    print "================================================"
    print "KEY:", key, T[key]
    print "================================================"
    subgraph = nx.Graph(T[key])
    # for clique in nx.find_cliques(subgraph):
    while True:
        clique = []
        for x in nx.find_cliques(subgraph):
            if len(x) > 1:
                clique = x
                break
        if len(clique) == 0:
            break
        sg = G.subgraph(clique)
        clique_str = ','.join(str(y) for y in sorted(clique))
        N.append(clique_str)
        G.remove_edges_from(sg.edges())
        subgraph.remove_edges_from(sg.edges())
        subgraph.number_of_edges()

        # of the edges I just removed.. did I create any singletons that need to be grammarred
        singletons = []
        for n in clique:
            if len(G.neighbors(n)) == 0:
                singletons.append(str(n))

        # grammarize these nodes
        if len(singletons) > 0:
            # find all the nonterminals they are incident to
            incident_nonterminals = mset()
            for x in singletons:
                local_incident_nonterminals = []
                # need to be smart about double counting.
                for n in N:
                    if x in n.split(','):
                        local_incident_nonterminals.append(n)
                incident_nonterminals = incident_nonterminals | mset(local_incident_nonterminals)
            incident_nonterminals = list(incident_nonterminals.elements())

            # find the outer nodes (all nodes in incident_nonterminals - singletons)
            outer_nodes = set()
            for x in incident_nonterminals:
                for y in x.split(","):
                    outer_nodes.add(y)
            for x in singletons:
                outer_nodes.remove(x)

            # compression step
            #if len(outer_nodes) > 3:
            #    print "Outer nodes: " + str(len(outer_nodes))
            #    G.remove_nodes_from(singletons)
            #    continue

            # starting condition
            if len(outer_nodes) == 0:
                print "starting..."
                outer_nodes.add("S")

            rhs = []
            for x in incident_nonterminals:
                rhs.append(x)
            # we can merge a step if the removal of a single terminal edge caused the singleton
            if clique_str in incident_nonterminals:
                rhs.remove(clique_str)
                for x in sg.edges():
                    rhs.append(x)

            # draw a nonterminal between outernodes
            outer_nodes_str = ','.join(str(y) for y in sorted(outer_nodes))
            N.append(outer_nodes_str)

            insert_rule(outer_nodes_str, rhs)
            print "    SINGLETONS", singletons
            print "   ", outer_nodes_str, rhs

            # remove the replaced nonterminals from the set
            for x in incident_nonterminals:
                if x in N:
                    N.remove(x)

            # remove singletons
            G.remove_nodes_from(singletons)
        else:
            insert_rule(clique_str, sg.edges())
            print "    NO SINGLETONS"
            print "   ", clique_str, sg.edges()

print "PRODUCTION RULES:"
for x in production_rules:
    print x + " => "
    for y in production_rules[x]:
        print "\t" + y + ": " + str(production_rules[x][y])
