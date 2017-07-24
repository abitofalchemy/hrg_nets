__authors__ = 'saguinag,tweninge,dchiang'
__contact__ = '{authors}@nd.edu'
__version__ = "0.1.0"

# scale_sampled_graph.py  

# VersionLog:
# 0.1.0 Initial state; 

import math
import re

import networkx as nx
import pandas as pd
import david as pcfg
import graph_sampler as gs
# import net_metrics as metrics
import product
import tw_karate_chop as tw
from gg import binarize, graph_checks, grow, graphical_degree_sequence

#import matplotlib
#matplotlib.use('agg')
from matplotlib import pyplot as plt
plt.switch_backend('agg')
import matplotlib.pyplot as plt
#plt.style.use('ggplot')

prod_rules = {}
debug = False


def learn_grammars_production_rules(input_graph):
	G = input_graph
	print G.number_of_nodes()
	print G.number_of_edges()
	num_nodes = G.number_of_nodes()

	G.remove_edges_from(G.selfloop_edges())
	giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
	G = nx.subgraph(G, giant_nodes)

	graph_checks(G)

	print
	print "--------------------"
	print "-Tree Decomposition-"
	print "--------------------"

	if num_nodes >= 500:
		for Gprime in gs.rwr_sample(G, 2, 100):
			T = tw.quickbb(Gprime)
			root = list(T)[0]
			T = tw.make_rooted(T, root)
			T = binarize(T)
			root = list(T)[0]
			root, children = T
			tw.new_visit(T, G, prod_rules)
	else:
		T = tw.quickbb(G)
		root = list(T)[0]
		T = tw.make_rooted(T, root)
		T = binarize(T)
		root = list(T)[0]
		root, children = T
		tw.new_visit(T, G, prod_rules)

	# return
	return prod_rules

def draw_degree_probability_distribution(orig_g, phrg, kpgm, clgm,nbrnodes='',gname='',axs=None):
	with open('../Results/deg_prob_dist_{}_{}.txt'.format(gname,nbrnodes), 'w') as f:
		d = orig_g.degree()
		n = orig_g.number_of_nodes()
		df = pd.DataFrame.from_dict(d.items())
		df.columns = ['v', 'k']
		gb = df.groupby(by=['k'])

		if axs is None:
			f, axs = plt.subplots(1, 1, figsize=(1.6 * 6, 1 * 6))
		x = gb.count().index.values
		y = gb.count().values / float(n)
		print 'orig graph'
		axs.plot(x, y, '-o', color='k')  # plot distribution of original graph

		f.write('# original graph\n')
		for i in range(len(y)):
			f.write('({}, {})\n'.format(x[i], y[i]))

		col_names = []
		multigraph_df = pd.DataFrame()

		for mG in [phrg,kpgm,clgm]:
			for i, hstar in enumerate(mG):
				d = hstar.degree()
				n = len(d)
				df = pd.DataFrame.from_dict(d.items())
				gb = df.groupby(by=[1])
				col_names.append('H*_{}'.format(i))
				multigraph_df = pd.concat([multigraph_df, gb.count()], axis=1)  # Appends to bottom new DFs

			cdf = multigraph_df / float(n)

			# print type (cdf)
			cdf.columns = col_names

			# cdf.plot(ax=axs, colormap='Greens_r', marker='o', linestyle=':', alpha=0.8)

			axs.plot(cdf.index, cdf.mean(axis=1), ':.')
			# one sigma
			# axs.fill_between(cdf.index, cdf.mean(axis=1) -cdf.std(axis=1), cdf.mean(axis=1) +cdf.std(axis=1), alpha=0.2)

			#   ## special case
			f.write('# new graphs set')
			yy = cdf.mean(axis=1).values
			for i,mu in enumerate(yy):
				f.write('({}, {})\n'.format(cdf.index[i], mu))


		axs.set_ylabel(r"$p(k)$")
		axs.set_xlabel(r"degree, $k$")
		# axs.patch.set_facecolor('None')
		# # special case:
		# axs.set_ylim(orig_floor*0.9,1.0)
		axs.set_xlim(0.9,10**2)

		axs.grid(True, which='both')
		axs.spines['left'].set_color('#B0C4DE')
		axs.yaxis.tick_left()
		axs.spines['bottom'].set_color('#B0C4DE')
		axs.xaxis.tick_bottom()
		axs.set_yscale('log')
		axs.set_xscale('log')

# ~~~~~~~~~~~~~~~~
# * Main - Begin *


# seed graph 
# G = nx.karate_club_graph()
#G = nx.read_edgelist("../demo_graphs/as20000102.txt")  #
G = nx.read_edgelist("../demo_graphs/out.moreno_propro_propro", comments="%")  # load_graphs("KarateClub")



n = G.number_of_nodes()
degree_sequence = G.degree().values()

prod_rules = learn_grammars_production_rules(G)

print
print "--------------------"
print "- Production Rules -"
print "--------------------"

for k in prod_rules.iterkeys():
	# print k
	s = 0
	for d in prod_rules[k]:
		s += prod_rules[k][d]
	for d in prod_rules[k]:
		prod_rules[k][d] = float(prod_rules[k][d]) / float(s)  # normailization step to create probs not counts.
		# print '\t -> ', d, prod_rules[k][d]
#

rules = []
id = 0
for k, v in prod_rules.iteritems():
	sid = 0
	for x in prod_rules[k]:
		rhs = re.findall("[^()]+", x)
		rules.append(("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x]))
		# print ("r%d.%d" % (id, sid), "%s" % re.findall("[^()]+", k)[0], rhs, prod_rules[k][x])
		sid += 1
	id += 1

g = pcfg.Grammar('S')
for (id, lhs, rhs, prob) in rules:
	g.add_rule(pcfg.Rule(id, lhs, rhs, prob))

print "Starting max size"

print '(n=nodes,m=edges):', G.number_of_nodes(), G.number_of_edges()
n = G.number_of_nodes()

f, axs = plt.subplots(1, 1, figsize=(1.6 * 6., 1 * 6.))

target_nodes_lst = [.125, .25, .5, 1, 2, 4, 8, 16, 32]

multiGraphs   = []
chungluGraphs = []
kronGraphs    = []
arr_deg_sequences = []

for target_nodes in target_nodes_lst:
	target_nodes = int(n*target_nodes)

	g.set_max_size(target_nodes) # target_nodes or multiple of n
	print "Done with max size"

	for run in range(1):
		#rule_list = g.sample(target_nodes)
		# PHRG
		#hstar = grow(rule_list, g)[0]
		print run,':', target_nodes

		# CLGM
		z = graphical_degree_sequence(target_nodes)
                print z
		arr_deg_sequences.append(z)
		clgm = nx.expected_degree_graph(z)
                #ddbreak
		# KPGM -
		#k = int(math.log(target_nodes, 2))
		# from: i:/data/saguinag/Phoenix/demo_graphs/karate.txt
		# Karate Club
		# P = [[0.9999,0.661],[0.661,     0.01491]]
		# Interent: autonomous systems
		#P = [[0.9523, 0.585], [0.585, 0.05644]]
		#kpgm = product.kronecker_random_graph(k, P)
		#for u, v in kpgm.selfloop_edges():
		#	kpgm.remove_edge(u, v)


		#multiGraphs.append(hstar)
		chungluGraphs.append(clgm)
		#kronGraphs.append(kpgm)

	#print 'H*(m,n)',  hstar.number_of_nodes(), hstar.number_of_edges()
	#print 'KP(m,n):', kpgm.number_of_nodes(), kpgm.number_of_edges()
	print 'CL(m,n):', clgm.number_of_nodes(), clgm.number_of_edges()
	

	# metrics.draw_degree_probability_distribution(G, multiGraphs, axs, 'b', 'b', gname='PHRG' + str(target_nodes))
	# metrics.draw_degree_probability_distribution(G, chungluGraphs, axs, 'r', 'r', gname='ChLu' + str(target_nodes))
	# metrics.draw_degree_probability_distribution(G, kronGraphs, axs, 'g', 'g', gname='Kron' + str(target_nodes))
#draw_degree_probability_distribution(G, multiGraphs, kronGraphs,chungluGraphs,
#                                     nbrnodes=str(target_nodes),gname='AS4x', axs=axs)
# end of for loop


#plt.legend(labels=['Orig Graph', 'PHRG', 'KronProd', 'CLGM'])
#axs.set_ylim([0.007,1.])
plt.savefig('outfig', bb_inches='tight', dpi=f.get_dpi())
plt.close()

# write arr_deg_sequences to disk
df = pd.DataFrame(arr_deg_sequences)
df.to_csv('../Results/arr_deg_sequences_proteins',sep='\t', header=False, index=False)

print 'Done'
# print '(m,n):', clgraph.number_of_nodes(), clgraph.number_of_edges()
