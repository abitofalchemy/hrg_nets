import random
import networkx as nx
import random
import datetime
from collections import Counter as mset
from collections import namedtuple
import re


class Markov(object):
    def __init__(self, order_size):
        self.order_size = order_size
        self.cache = {}

    def insert(self, history, entry):
        if self.order_size != len(history):
            raise Exception("History does not match Markov size.")
        if self.order_size == 1:
            if history[0] not in self.cache:
                self.cache[history[0]] = []
            self.cache[history[0]].append(entry)
        else:
            if history[0] not in self.cache:
                self.cache[history[0]] = Markov(self.order_size - 1)
            self.cache[history[0]].insert(history[1:], entry)

    def fetch(self, history):
        if self.order_size != len(history):
            raise Exception("History does not match Markov size.")
        if self.order_size == 1:
            return self.cache[history[0]]
        else:
            return self.cache[history[0]].fetch(history[1:])

    def get_size(self):
        return self.order_size

    def dump(self):
        if self.order_size == 1:
            return {k: v for k, v in self.cache.items()}
        return {k: v.dump() for k, v in self.cache.items()}

    def print_dump(self, indent):
        new_indent = '    '
        if self.order_size == 1:
            for k, v in self.cache.items():
                print "%s'%s':" % (indent, k)
                print "%s%s" % (indent + new_indent, v)
        else:
            for k, v in self.cache.items():
                print "%s'%s':" % (indent, k)
                v.print_dump(indent + new_indent)


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


'''
model = Markov(3)

model.insert(['a', 'b', 'c'], "foo")
model.insert(['a', 'b', 'c'], "bar")
model.insert(['a', 'b', 'c'], "baz")
model.insert(['a', 'bc', 'd'], "foo")

print model.dump()
'''


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

production_rules = {}

G, T = load_koblenz_quad("../demo_graphs/haggle_contact_koblenz.txt")

print "Source Graph nodes: %d" % G.number_of_nodes()
print "Source Graph edges: %d" % G.number_of_edges()

N = []
GT = nx.DiGraph()
R = []
seen = set()
rhs_dict = {}

uncompressed_rule = []
created_in_rule = {}


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
    total_in_cluser = i

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
    #print lhs_str + " => " + rhs_str

    if lhs_str in production_rules:
        production_rules[lhs_str][rhs_str] += 1
    else:
        rhs_mset = mset()
        rhs_mset[rhs_str] += 1
        production_rules[lhs_str] = rhs_mset
    uncompressed_rule.append((lhs, lhs_str, (total_in_cluser, rhs_str)))
    return len(uncompressed_rule) - 1




for key in sorted(T.keys()):
    # backwards version
    #print "================================================"
    #print "KEY:", key, T[key]
    #print "================================================"
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
                #print "starting..."
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

            rule_number = insert_rule(outer_nodes_str, rhs)
            #print "    SINGLETONS", singletons
            #print "   ", outer_nodes_str, rhs

            for created in rhs:
                if isinstance(created, str) and created not in created_in_rule:
                    created_in_rule[created] = rule_number

            # remove the replaced nonterminals from the set
            for x in incident_nonterminals:
                if x in N:
                    N.remove(x)

            # remove singletons
            G.remove_nodes_from(singletons)
        else:
            rule_number = insert_rule(clique_str, sg.edges())
            #print "    NO SINGLETONS"
            #print "   ", clique_str, sg.edges()

'''
print "PRODUCTION RULES:"
for x in production_rules:
    print x + " => "
    for y in production_rules[x]:
        print "\t" + y + ": " + str(production_rules[x][y])
'''

#print uncompressed_rule[::-1]
#print created_in_rule


def backtrack(start, depth):
    if depth:
        if start in created_in_rule:
            return backtrack(uncompressed_rule[created_in_rule[start]][0], depth - 1) + [uncompressed_rule[created_in_rule[start]][2]]
        else:
            return backtrack('', depth - 1) + [(0, '')]
    return []


model = Markov(3)
for rule in uncompressed_rule:
    model.insert([base_rule for count, base_rule in backtrack(rule[0], model.get_size())], rule[2])

model.print_dump('')


# print [num_to_word(x) for x in xrange(max([rule[2][0]  for rule in uncompressed_rule]))]

Clique = namedtuple('Clique', ['rule', 'history', 'nids'])
cliques = [Clique(rule='', history=None, nids=[])]
node_count = -1
edges = []


def get_generation_history(clique, depth):
    if depth:
        if clique.history is None or clique.history >= len(cliques):
            return get_generation_history(clique, depth - 1) + ['']
        return get_generation_history(cliques[clique.history], depth - 1) + [clique.rule]
    return []


current_clique = 0
while current_clique < len(cliques):
    clique = cliques[current_clique]
    history = get_generation_history(clique, model.get_size())
    possibilities = [lhs for size, lhs in model.fetch(history) if size == len(clique.nids)]
    rhs = random.choice(possibilities)
    ids_used = {num_to_word(x): y for x, y in zip(xrange(len(clique.nids)), clique.nids)}
    for ids, type in [(rule[0].split(','), rule[1]) for rule in [rule.split(':') for rule in re.findall("[^()]+", rhs)]]:
        for id in ids:
            if id not in ids_used:
                node_count += 1
                ids_used[id] = node_count
        if type == 'N':
            # This is a non-terminal.  Create a clique and add to the list
            new_clique = Clique(rule=rhs, history=current_clique, nids=[ids_used[id] for id in ids])
            cliques.append(new_clique)
        else:
            # This is a terminal
            edges.append((ids_used[ids[0]], ids_used[ids[1]]))
    current_clique += 1

print "Nodes: %d" % (node_count + 1)
print "Edges: %d" % len(edges)
print "Cliques computed: %d" % len(cliques)
