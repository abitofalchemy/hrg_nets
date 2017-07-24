import random
import networkx as nx
import random
import datetime
from collections import Counter as mset
from collections import namedtuple
import re
import data.graphs as graphs
import sys
from time import time


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


production_rules = {}

buckets = 100
G, EdgeTimes = graphs.HepPh()
T = graphs.groupTimes(EdgeTimes, buckets)
print "Bucket Counts:"
for i in range(buckets):
    print "\t", str(i) + ":", len(T[i])

print "Source Graph nodes: %d" % G.number_of_nodes()
print "Source Graph edges: %d" % G.number_of_edges()

N = []
GT = nx.DiGraph()
R = []
seen = set()
rhs_dict = {}

uncompressed_rule = []
created_in_rule = {}
created_in_context = {}


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
    total_in_cluster = i

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
    uncompressed_rule.append((lhs, lhs_str, (total_in_cluster, rhs_str)))
    return len(uncompressed_rule) - 1



start_time = last_time = time()
original_edge_count = G.number_of_edges()
character_count = len(str(original_edge_count))
previous_outstr_len = 0
for key in sorted(T.keys(), reverse=True):
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
        now = time()
        if now - last_time > 5:
            current_edges = G.number_of_edges()
            if not original_edge_count == current_edges:
                predicted_time_total = (now - start_time) * original_edge_count / (original_edge_count - current_edges)
                predicted_time_left = predicted_time_total - (now - start_time)
                outstr = "Predicted time remaining: " + str(datetime.timedelta(seconds=predicted_time_left)).ljust(28) + "Edges remaining: " + str(current_edges).rjust(character_count) + "/" + str(original_edge_count) + '\r'
                print outstr.ljust(previous_outstr_len),
                sys.stdout.flush()
                previous_outstr_len = len(outstr)
            last_time = now
        sg = G.subgraph(clique)
        clique_str = ','.join(str(y) for y in sorted(clique))
        #print "clique_str", clique_str
        N.append(clique_str)
        G.remove_edges_from(sg.edges())
        subgraph.remove_edges_from(sg.edges())

        # of the edges I just removed.. did I create any singletons that need to be grammarred
        singletons = []
        for n in clique:
            if len(G.neighbors(n)) == 0:
                singletons.append(str(n))
        #print "singletons: ", singletons

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
            #print "incident_nonterminals", incident_nonterminals

            # find the outer nodes (all nodes in incident_nonterminals - singletons)
            outer_nodes = set()
            for x in incident_nonterminals:
                for y in x.split(","):
                    outer_nodes.add(y)
            for x in singletons:
                outer_nodes.remove(x)
            #print "outer_nodes:", outer_nodes

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

            # Figure out the context in which each singleton is created
            # For each singleton, an entry will be added containing the
            # number of terminals ("T2" = two terminal connections) as well
            # as non-terminal groups ("N3" = a group of 3 non-terminals)
            # that the singleton was connected to upon generation.
            for cluster in [x for x in rhs if isinstance(x, str)]:
                cluster_array = cluster.split(',')
                cluster_set = set(cluster_array)
                cluster_int_set = set([int(y) for y in cluster_array])
                created_in_context[cluster] = ','.join(["T%d" % len([x for x in rhs if isinstance(x, tuple) and len(cluster_int_set.intersection(x))])] + sorted(["N%d" % len(x.split(',')) for x in rhs if isinstance(x, str) and len(cluster_set.intersection(x.split(',')))]))

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


def backtrack_context(start, depth):
    if depth:
        if start in created_in_rule:
            return backtrack_context(uncompressed_rule[created_in_rule[start]][0], depth - 1) + [created_in_context[start]]
        else:
            return backtrack_context('', depth - 1) + ['']
    return []


model = Markov(3)
for rule in uncompressed_rule:
    model.insert(backtrack_context(rule[0], model.get_size()), rule[2])

#model.print_dump('')


# print [num_to_word(x) for x in xrange(max([rule[2][0]  for rule in uncompressed_rule]))]

def get_generation_history(clique, depth, cliques):
    if depth:
        if clique.history is None or clique.history >= len(cliques):
            return get_generation_history(clique, depth - 1, cliques) + ['']
        return get_generation_history(cliques[clique.history], depth - 1, cliques) + [clique.rule]
    return []


def generate(model):
    Clique = namedtuple('Clique', ['rule', 'history', 'nids'])
    cliques = [Clique(rule='', history=None, nids=[])]
    node_count = -1
    edges = []

    current_clique = 0
    while current_clique < len(cliques):
        clique = cliques[current_clique]
        history = get_generation_history(clique, model.get_size(), cliques)
        #print "H", history
        #print "C", clique
        #print "A", model.fetch(history)
        possibilities = [lhs for size, lhs in model.fetch(history) if size == len(clique.nids)]
        #print "P", possibilities
        rhs = random.choice(possibilities)
        ids_used = {num_to_word(x): y for x, y in zip(xrange(len(clique.nids)), clique.nids)}
        rhs_explode = [(rule[0].split(','), rule[1], rule[0]) for rule in [rule.split(':') for rule in re.findall("[^()]+", rhs)]]
        #print 'E', rhs_explode
        for ids, type, cluster in rhs_explode:
            for id in ids:
                if id not in ids_used:
                    node_count += 1
                    ids_used[id] = node_count
            if type == 'N':
                # This is a non-terminal.  Create a clique and add to the list
                temp_set = [x for x in rhs_explode if len(set(ids).intersection(x[0]))]
                identifier = ','.join(["T%d" % len([x for x in temp_set if x[1] == 'T'])] + sorted(["N%d" % len(x[0]) for x in temp_set if x[1] == 'N']))
                new_clique = Clique(rule=identifier, history=current_clique, nids=[ids_used[id] for id in ids])
                cliques.append(new_clique)
            else:
                # This is a terminal
                edges.append((ids_used[ids[0]], ids_used[ids[1]]))
        current_clique += 1
    return edges, node_count, cliques

edges, node_count, cliques = generate(model)

print "Nodes: %d" % (node_count + 1)
print "Edges: %d" % len(edges)
print "Cliques computed: %d" % len(cliques)

print "Nodes", "Edges", "Cliques Computed"
for i in range(20):
    edges, node_count, cliques = generate(model)
    print str(node_count).ljust(5), str(len(edges)).ljust(5), str(len(cliques)).ljust(16)

