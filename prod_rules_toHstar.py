import shelve
import os
import re
import networkx as nx
import tw_karate_chop as tw
import net_metrics as metrics
import graph_sampler as gs
import david as pcfg

# 
# # Load production rules
# ######################## 
shelf = shelve.open("../Results/production_rules_dict.shl.db") # the same filename that you used before, please
prod_rules = shelf["karate"]
shelf.close()

rules = []
id = 0
for k, v in prod_rules.iteritems():
    sid = 0
    for x in prod_rules[k]:
        rhs = re.findall("[^()]+", x)
        rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
        #print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])
        sid += 1
    id += 1

g = pcfg.Grammar('S')
for (id, lhs, rhs, prob) in rules:
    g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

print "Starting max size"

G = nx.karate_club_graph()

g.set_max_size(G.number_of_edges())

print "Done with max size"

graphletG = []
graphletH = []
multiGraphs =[]

rule_list = g.sample(G.number_of_edges())
#print rule_list#type(rule_list),'rule_list', len(rule_list)
import hrgm_ns_sal as hrgg

hrgg.hstar = grow(rule_list, g)[0]

print hstar.number_of_nodes()
print hstar.number_of_edges()

import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

f, axs = plt.subplots(1, 2, figsize=(1.6 * 6, 1 * 6))

nx.draw_networkx(G,ax=axs[0])
nx.draw_networkx(hstar,ax=axs[1])

plt.savefig('outfig', bb_inches='tight')


