import random

import math
import collections
import tree_decomposition as td
import create_production_rules as pr
import graph_sampler as gs
import stochastic_growth
import probabilistic_growth
import net_metrics
import matplotlib.pyplot as plt
import product

import networkx as nx
import numpy as np
import snap

#G = snap.GenRndGnm(snap.PUNGraph, 10000, 5000)

#G = nx.grid_2d_graph(4,4)

#line
#G = nx.Graph()
#G.add_edge(1, 2)
#G.add_edge(2, 3)
#G.add_edge(3, 4)
#G.add_edge(4, 5)
#G.add_edge(5, 6)
#G.add_edge(6, 7)
#G.add_edge(7, 8)
#G.add_edge(8, 9)
#G.add_edge(9, 10)
#G.add_edge(10, 1) #circle

#G = nx.star_graph(6)
#G = nx.ladder_graph(10)


#G = nx.karate_club_graph()

#nx.write_edgelist((G.to_directed()), '../demo_graphs/karate.txt', comments="#", delimiter=' ', data=False)
#exit()
#G = nx.barabasi_albert_graph(1000,3)
#G = nx.connected_watts_strogatz_graph(200,8,.2)

#G = nx.read_edgelist("../demo_graphs/as20000102.txt")
G = nx.read_edgelist("../demo_graphs/CA-GrQc.txt")
#G = nx.read_edgelist("../demo_graphs/Email-Enron.txt")
#G = nx.read_edgelist("../demo_graphs/Brightkite_edges.txt")
G= list(nx.connected_component_subgraphs(G))[0]



##board example

#G = nx.Graph()
#G.add_edge(1, 2)
#G.add_edge(2, 3)
#G.add_edge(2, 4)
#G.add_edge(3, 4)
#G.add_edge(3, 5)
#G.add_edge(4, 6)
#G.add_edge(5, 6)
#G.add_edge(1, 5)



# print G.number_of_nodes()

num_nodes = G.number_of_nodes()
print num_nodes

print
print "--------------------"
print "------- Edges ------"
print "--------------------"

num_edges = G.number_of_edges()
print num_edges

#print
#print "--------------------"
#print "------ Cliques -----"
#print "--------------------"

#print list(nx.find_cliques(G))


if not nx.is_connected(G):
    print "Graph must be connected";
    exit()

G.remove_edges_from(G.selfloop_edges())
if G.number_of_selfloops() > 0:
    print "Graph must be not contain self-loops";
    exit()

Ggl = gs.subgraphs_cnt(G,100)

setlendf = []

if num_nodes>400:
    #for i in range(0,10):
    #    setlen = []
    #    for i in range(10,510, 20):
    for Gprime in gs.rwr_sample(G, 10, 500):
        pr.prod_rules = {}
        T = td.quickbb(Gprime)
        prod_rules = pr.learn_production_rules(Gprime, T)
#        setlen.append(len(rule_probabilities))
        print prod_rules

else:
    T = td.quickbb(G)
    prod_rules = pr.learn_production_rules(G, T)

print "Rule Induction Complete"


exit()

Gergm = []
Gergmgl = []


for run in range(1, 3):
    f = open('../demo_graphs/ergm_sim/enron/data '+str(run)+' .csv', 'r')
    E=nx.Graph()
    header = 0
    for line in f:
        line=line.rstrip()
        if header == 0:
            header+=1
            continue
        c = line.split("\t")
        if(len(c) is not 3): continue
        E.add_edge(int(c[1]),int(c[2]))
        if int(c[1]) > num_nodes or int(c[2]) > num_nodes:
            continue
    Gergm.append(E)
    print "G ergm iteration " + str(run) + " of 20"
    Gergmgl.append(gs.subgraphs_cnt(E,50))

k = int(math.floor(math.log(num_nodes, 10)))
P = [[.9716,.658],[.5684,.1256]] #karate
P = [[.8581,.5116],[.5063,.2071]] #as20000102
#P = [[.7317,.5533],[.5354,.2857]] #dblp
#P = [[.9031,.5793],[.5051,.2136]] #ca-grqc
#P = [[.9124,.5884],[.5029,.2165]] #enron
P = [[.8884,.5908],[.5628,.2736]] #brightkite
Gkron = product.kronecker_random_graph(k,P).to_undirected()

print("GKron finished")

sum = .9716+.5382+.5684+.1256 #karate
#sum = .8581+.5116+.5063+.2071 #as20000102
#sum = .7317+.5533+.5354+.2857 # dblp
#sum = .9031+.5793+.5051+.2136 #ca-grqc
#sum = .9124+.5884+.5029+.2165 #enron
sum = .8884+.5908+.5628+.2736 #brightkite
GRmatSNAP = snap.GenRMat(num_nodes, num_edges, P[0][0]/sum, P[0][1]/sum, P[1][0]/sum)
GRmat = nx.Graph()
for EI in GRmatSNAP.Edges():
    GRmat.add_edge(EI.GetSrcNId(), EI.GetDstNId())

print("GRMAT finished")
GRmatgl = gs.subgraphs_cnt(GRmat,100)

n_distribution = {}
Gstar = []
Dstar = []
Gstargl = []
for run in range(0, 20):
    nG, nD = stochastic_growth.grow(prod_rules, num_nodes/10,0)#num_nodes/50)
    Gstar.append(nG)
    Dstar.append(nD)
    Gstargl.append(gs.subgraphs_cnt(nG,100))
    #Gstar.append(probabilistic_growth.grow(rule_probabilities,prod_rule_set, num_nodes))
    print "G* iteration " + str(run) + " of 20"

print(nD)

print ""
print "G* Samples Complete"

label = "AS"

net_metrics.draw_graphlet_plot(Ggl, Gstargl, Gergmgl, Gkron, GRmatgl, label, plt.figure())
exit()
net_metrics.draw_diam_plot(G, Dstar, Gergm, Gkron, GRmat, label, plt.figure())
net_metrics.draw_degree_rank_plot(G, Gstar, Gergm, Gkron, GRmat, label, plt.figure())
#net_metrics.draw_scree_plot(G, Gstar, label, ax1)
net_metrics.draw_network_value(G, Gstar, Gergm, Gkron, GRmat, label, plt.figure())
net_metrics.draw_hop_plot(G, Gstar, Gergm, Gkron, GRmat, label, plt.figure())

#ax1.plot(ef.mean().index, ef.mean()[1],'b')

net_metrics.save_plot_figure_2disk()