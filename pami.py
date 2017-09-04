#!/usr/bin/env python
__version__="0.1.0"
__author__ = 'tweninge'
__author__ = 'saguinag'


# Version: 0.1.0 Forked from cikm_experiments.py
#
# ToDo:
#


import subprocess, shlex
from threading import Timer
import math
import shelve 
import time

import traceback
import argparse
import os#, sys
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

import networkx as nx
import numpy as np

import nu_metrix as metrics
import HRG
import PHRG
import product
import pprint as pp

from load_edgelist_from_dataframe import Pandas_DataFrame_From_Edgelist
from salPHRG import grow_graphs_using_krongen


#def run(cmd, timeout_sec):
#		proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
#														stderr=subprocess.PIPE)
#		kill_proc = lambda p: p.kill()
#		timer = Timer(timeout_sec, kill_proc, [proc])
#		try:
#				timer.start()
#				stdout, stderr = proc.communicate()
#		finally:
#				timer.cancel()
def run(cmd, timeout_sec):
		proc = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
														stderr=subprocess.PIPE)
		kill_proc = lambda p: p.kill()
		timer = Timer(timeout_sec, kill_proc, [proc])
		try:
				timer.start()
				stdout, stderr = proc.communicate()
		finally:
				timer.cancel()

#def kronfit(G):
#	"""
#	Notes:
#	23May16: changes to handle kronfit
#	"""
#
#	from os import environ
#
#	with open('tmp.txt', 'w') as tmp:
#			for e in G.edges():
#					tmp.write(str(e[0]) + ' ' + str(e[1]) + '\n')
#
#	if environ['HOME'] == '/home/saguinag':
#		args = ("/home/saguinag/Software/Snap-3.0/examples/kronfit/kronfit", "-i:tmp.txt","-n0:2", "-m:\"0.9 0.6; 0.6 0.1\"", "-gi:5")
#	elif environ['HOME'] == '/Users/saguinag':
#		args = ("/Users/saguinag/ToolSet/Snap-3.0/examples/kronfit/kronfit", "-i:tmp.txt","-n0:2", "-m:\"0.9 0.6; 0.6 0.1\"", "-gi:5")
#	else:
#		args = ('./kronfit.exe -i:tmp.txt -n0:2 -m:"0.9 0.6; 0.6 0.1" -gi:5')
#	#print (args)
#	#args = args.split()
#	#options = {k: True if v.startswith('-') else v
#	#			 for k,v in zip(args, args[1:]+["--"]) if k.startswith('-')}
#
#	#proc = subprocess.Popen(shlex.split(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#	"""
#	proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#	kill_proc = lambda p: p.kill()
#	timer = Timer(10, kill_proc, [proc])
#	try:
#			timer.start()
#			output, stderr = proc.communicate()
#	finally:
#			timer.cancel()
#
#	print "out"
#	"""
#
#	kronout = ""
#	while not kronout:
#		popen = subprocess.Popen(args, stdout=subprocess.PIPE)
#		popen.wait()
#		output = popen.stdout.read()
#
#		top = []
#		bottom = []
#		kronout = output.split('\n')


def kronfit(G):
	"""

	Notes:
	23May16: changes to handle kronfit
	"""

	from os import environ
	if G.name is None:
		tsvGraphName = "/tmp/tmp.tsv"
	else:
		tsvGraphName = "/tmp/{}kpgraph.tsv".format(G.name)

	with open(tsvGraphName, 'w') as tmp:
			for e in G.edges():
					tmp.write(str(e[0]) + ' ' + str(e[1]) + '\n')

	if environ['HOME'] == '/home/saguinag':
		args = ("time/bin/linux/kronfit", "-i:{}".format(tsvGraphName),
				"-n0:2", "-m:\"0.9 0.6; 0.6 0.1\"", "-gi:5")
	elif environ['HOME'] == '/Users/saguinag':
		args = ("time/bin/macos/kronfit", "-i:{}".format(tsvGraphName),
				"-n0:2", "-m:\"0.9 0.6; 0.6 0.1\"", "-gi:5")
	else:
		args = ('./kronfit.exe -i:tmp.txt -n0:2 -m:"0.9 0.6; 0.6 0.1" -gi:5')
	#print (args)
	#args = args.split()
	#options = {k: True if v.startswith('-') else v
	#			 for k,v in zip(args, args[1:]+["--"]) if k.startswith('-')}

	#proc = subprocess.Popen(shlex.split(args), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	"""
	proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	kill_proc = lambda p: p.kill()
	timer = Timer(10, kill_proc, [proc])
	try:
			timer.start()
			output, stderr = proc.communicate()
	finally:
			timer.cancel()

	print "out"
	"""
	
	kronout = ""
	while not kronout:
		popen = subprocess.Popen(args, stdout=subprocess.PIPE)
		popen.wait()
		output = popen.stdout.read()

		top = []
		bottom = []
		kronout = output.split('\n')

	for i in range(0, len(kronout)):
			if kronout[i].startswith("FITTED PARAMS"):
					top = kronout[i + 1].split()
					top = [float(j) for j in top]
					bottom = kronout[i + 2].split()
					bottom = [float(j) for j in bottom]
					break

	if not len(top):
		print 'top:',top,'bottom',bottom
		top = [0,0]
		bottom = [0,0]

	top[1] = bottom[0] = (top[1] + bottom[0]) / 2	# make symmetric by taking average
	fitted = [top, bottom]
	return fitted


def gcd():
		num_nodes = 1000

		ba_G = nx.barabasi_albert_graph(num_nodes, 3)
		er_G = nx.erdos_renyi_graph(num_nodes, .1)
		ws_G = nx.watts_strogatz_graph(num_nodes, 8, .1)
		nws_G = nx.newman_watts_strogatz_graph(num_nodes, 8, .1)

		graphs = [ba_G, er_G, ws_G, nws_G]

		samples = 50

		for G in graphs:
				chunglu_M = []
				for i in range(0, samples):
						chunglu_M.append(nx.expected_degree_graph(G.degree()))

				HRG_M, degree = HRG.stochastic_hrg(G, samples)
				pHRG_M = PHRG.probabilistic_hrg(G, samples)
				kron_M = []
				rmat_M = []
				for i in range(0, samples):
						P = kronfit(G)
						k = math.log(num_nodes, 2)
						kron_M.append(product.kronecker_random_graph(int(math.floor(k)), P, directed=False))

				df_g = metrics.external_rage(G)
				gcd_chunglu = []
				gcd_phrg = []
				gcd_hrg = []
				gcd_kron = []
				for chunglu_M_s in chunglu_M:
						df_chunglu = metrics.external_rage(chunglu_M_s)
						rgfd = metrics.tijana_eval_rgfd(df_g, df_chunglu)
						gcm_g = metrics.tijana_eval_compute_gcm(df_g)
						gcm_h = metrics.tijana_eval_compute_gcm(df_chunglu)
						gcd_chunglu.append(metrics.tijana_eval_compute_gcd(gcm_g, gcm_h))
				for HRG_M_s in HRG_M:
						df_hrg = metrics.external_rage(HRG_M_s)
						rgfd = metrics.tijana_eval_rgfd(df_g, df_hrg)
						gcm_g = metrics.tijana_eval_compute_gcm(df_g)
						gcm_h = metrics.tijana_eval_compute_gcm(df_hrg)
						gcd_hrg.append(metrics.tijana_eval_compute_gcd(gcm_g, gcm_h))
				for pHRG_M_s in pHRG_M:
						df_phrg = metrics.external_rage(pHRG_M_s)
						rgfd = metrics.tijana_eval_rgfd(df_g, df_phrg)
						gcm_g = metrics.tijana_eval_compute_gcm(df_g)
						gcm_h = metrics.tijana_eval_compute_gcm(df_phrg)
						gcd_phrg.append(metrics.tijana_eval_compute_gcd(gcm_g, gcm_h))
				for kron_M_s in kron_M:
						df_kron = metrics.external_rage(kron_M_s)
						rgfd = metrics.tijana_eval_rgfd(df_g, df_kron)
						gcm_g = metrics.tijana_eval_compute_gcm(df_g)
						gcm_h = metrics.tijana_eval_compute_gcm(df_kron)
						gcd_kron.append(metrics.tijana_eval_compute_gcd(gcm_g, gcm_h))

				print gcd_chunglu
				print gcd_hrg
				print gcd_phrg
				print gcd_kron
				print
				print

def synth_plots():
		num_nodes = 100
		samples = 5

		chunglu_M = []
		kron_M = []
		HRG_M = []
		pHRG_M = []
		G_M = []

		for i in range(0,samples):
				##BA Graph
				G = nx.erdos_renyi_graph(num_nodes, .1)
				G_M.append(G)

				for i in range(0, samples):
						chunglu_M.append(nx.expected_degree_graph(G.degree().values()))

				HRG_M_s, degree = HRG.stochastic_hrg(G, samples)
				HRG_M = HRG_M + HRG_M_s
				pHRG_M_s = PHRG.probabilistic_hrg(G, samples)
				pHRG_M = pHRG_M + pHRG_M_s
				for i in range(0, samples):
						P = kronfit(G)
						k = math.log(num_nodes, 2)
						kron_M.append(product.kronecker_random_graph(int(math.floor(k)), P, directed=False))

		metrics.draw_network_value(G_M, chunglu_M, HRG_M, pHRG_M, kron_M)

# --< main >--

# print '~'*100
# #G = nx.read_edgelist("../demo_graphs/Email-Enron.txt")
# #G = nx.read_edgelist("../demo_graphs/com-dblp.ungraph.txt")
# #G = nx.read_edgelist("../demo_graphs/as20000102.txt")


#G = nx.read_edgelist("../demo_graphs/CA-GrQc.txt")
#G = nx.read_edgelist("../kpmg.el")
#G.name = 'kpmg'
## import pandas as pd
## dat = np.loadtxt('../demo_graphs/karate.txt',dtype=int, delimiter=r'\s')
## df = pd.DataFrame(dat)
## df.columns=['u','v']
## print df.head()
## print df.columns
##G = nx.read_edgelist("../demo_graphs/karate.txt")
#
##df_g = metrics.external_rage(G)
##gcm_g = metrics.tijana_eval_compute_gcm(df_g)
#Hstars = PHRG.probabilistic_hrg(G,num_samples=100)
#if 0:
#	print type(H)
#	print len(H)
#	for g in H:
#		#print g.number_of_nodes(), g.number_of_edges(), nx.average_degree(g)
#		print nx.info(g)
#
#	print
#	print nx.info(G)
#
#metricx = [ 'degree','hops', 'clust', 'assort', 'kcore','eigen','gcd']
##metricx = ['degree', 'gcd']
#metrics.network_properties([G], metricx, Hstars, name=G.name, out_tsv=True)
#print 'Done'
#exit()
#
##
##
## print (nx.info(G))
## print(nx.average_clustering(G))
#global_clust_coeff =[]
#gcd_values = []
#for h in H:
#	global_clust_coeff.append(nx.average_clustering(h))
#	df = metrics.external_rage(h)
#	gcm_h = metrics.tijana_eval_compute_gcm(df)
#	# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_g) # CONTROL
#	# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h) # CONTROL
#	gcd_values.append(metrics.tijana_eval_compute_gcd(gcm_g, gcm_h))
#
#print '\t', np.mean(global_clust_coeff)
#print '\t', np.mean(gcd_values), np.std(gcd_values)
#
#
#
## print df_g
##
#print '~'*40
# """
# import numpy as np
# is_grpahical = True
# seq = np.zeros(n)
# while is_graphical:
#		 seq = np.random.poisson(4.588, n)
#		 is_graphical = nx.is_valid_degree_sequence_havel_hakimi(seq.tolist())
# Gi = nx.expected_degree_graph(seq.tolist())
# """
#
# import numpy as np
# for i in [1,2,3,4,5,6,7,8,9,10]:
#
#		 #k = math.log(n, 2)
#		 #print type(G)
#		 #P = kronfit(G)
#		 #Gi = product.kronecker2_random_graph(int(math.ceil(k)), P, directed=False)
#
# #		Gi = PHRG.probabilistic_hrg(G, 1, n/10)
#		 Gi = nx.expected_degree_graph(G.degree().values())
#		 df = metrics.external_rage(Gi)
#		 gcm_h = metrics.tijana_eval_compute_gcm(df)
#		 s = metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
#		 with open('clgm_dblp_gcd_infinity.txt', 'a') as tmp:
#				 tmp.write("(" + str(i) + "," + str(s) + ')\n')
#		 #print str(i), str(s)
#
#		 G = Gi
#		 print '~'*100
# exit()
#
# nx.write_edgelist(G,"phrg10.txt")
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_kron_0.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_kron_1.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_kron_2.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_cl_0.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_cl_1.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_cl_2.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_cl_3.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_hrg_0.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_hrg_1.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_phrg_0.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_phrg_1.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_phrg_2.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_phrg_3.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_phrg_4.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
#
# exit()
#
# #G.remove_edges_from(G.selfloop_edges())
# #G = nx.barabasi_albert_graph(100,3)
# G = nx.karate_club_graph()
# #G = nx.read_edgelist("../demo_graphs/Email-Enron.txt")
# #G = nx.read_edgelist("../demo_graphs/com-dblp.ungraph.txt")
# #G = nx.read_edgelist("../demo_graphs/as20000102.txt")
# #G = nx.read_edgelist("../demo_graphs/CA-GrQc.txt")
# # power_G = nx.read_edgelist("../Phoenix/demo_graphs/power.txt")
# #G.remove_edges_from(G.selfloop_edges())
#
# #P = [[.8581, .5063], [.5063, .2071]]	# as20000102
# #P = [[.9124, .5029], [.5029, .2165]]	# enron
# P = [[.7317, .5354], [.5354, .2857]]
#
#	 # dblp

#
# #G = nx.read_edgelist("../demo_graphs/protein.txt")
# G = nx.karate_club_graph()
# n = G.number_of_nodes()
#
# #P = [[.7395, .5093], [.5093, .2189]]	# protein
# #P = [[.7305, .5077], [.5077, .2197]]	# karate
#
# df_g = metrics.external_rage(G)
# gcm_g = metrics.tijana_eval_compute_gcm(df_g)
#
# print gcm_g
#
# print '~'*100
# """
# import numpy as np
# is_grpahical = True
# seq = np.zeros(n)
# while is_graphical:
#		 seq = np.random.poisson(4.588, n)
#		 is_graphical = nx.is_valid_degree_sequence_havel_hakimi(seq.tolist())
# Gi = nx.expected_degree_graph(seq.tolist())
# """
#
# import numpy as np
# for i in [1,2,3,4,5,6,7,8,9,10]:
#
#		 #k = math.log(n, 2)
#		 #print type(G)
#		 #P = kronfit(G)
#		 #Gi = product.kronecker2_random_graph(int(math.ceil(k)), P, directed=False)
#
# #		Gi = PHRG.probabilistic_hrg(G, 1, n/10)
#		 Gi = nx.expected_degree_graph(G.degree().values())
#		 df = metrics.external_rage(Gi)
#		 gcm_h = metrics.tijana_eval_compute_gcm(df)
#		 s = metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
#		 with open('clgm_dblp_gcd_infinity.txt', 'a') as tmp:
#				 tmp.write("(" + str(i) + "," + str(s) + ')\n')
#		 #print str(i), str(s)
#
#		 G = Gi
#		 print '~'*100
# exit()
#
# nx.write_edgelist(G,"phrg10.txt")
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_kron_0.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_kron_1.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_kron_2.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_cl_0.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_cl_1.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_cl_2.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_cl_3.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_hrg_0.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_hrg_1.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_phrg_0.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_phrg_1.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_phrg_2.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_phrg_3.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
# Gi = (nx.read_edgelist("../demo_graphs/dblp_phrg_4.txt"))
# df = metrics.external_rage(Gi)
# gcm_h = metrics.tijana_eval_compute_gcm(df)
# print metrics.tijana_eval_compute_gcd(gcm_g, gcm_h)
#
#
# exit()
#
# #G.remove_edges_from(G.selfloop_edges())
# #G = nx.barabasi_albert_graph(100,3)
# G = nx.karate_club_graph()
# #G = nx.read_edgelist("../demo_graphs/Email-Enron.txt")
# #G = nx.read_edgelist("../demo_graphs/com-dblp.ungraph.txt")
# #G = nx.read_edgelist("../demo_graphs/as20000102.txt")
# #G = nx.read_edgelist("../demo_graphs/CA-GrQc.txt")
# # power_G = nx.read_edgelist("../Phoenix/demo_graphs/power.txt")
# #G.remove_edges_from(G.selfloop_edges())
#
# #P = [[.8581, .5063], [.5063, .2071]]	# as20000102
# #P = [[.9124, .5029], [.5029, .2165]]	# enron
# P = [[.7317, .5354], [.5354, .2857]]
#
#	 # dblp
# P = [[.9124, .5029], [.5029, .2165]]	# enron
# #P = [[.7317, .5354], [.5354, .2857]]	# dblp
# #P = [[.9031, .5051], [.5051, .2136]]	# ca-grqc
#
# num_nodes = G.number_of_nodes()
# samples = 10
#
# chunglu_M = []
# kron_M = []
# HRG_M = []
# pHRG_M = []
# G_M = []
#
# """"
# for i in range(0, 5):

# """"
# for i in range(0, 5):
#		 Gk = nx.expected_degree_graph(G.degree().values())
#		 Gk.remove_edges_from(Gk.selfloop_edges())
#
#		 with open('dblp_cl_'+str(i)+'.txt', 'w') as tmp:
#				 for e in Gk.edges():
#						 tmp.write(str(e[0]) + ' ' + str(e[1]) + '\n')
#
#
# HRG_M_s, degree = HRG.stochastic_hrg(G, 2)
# HRG_M = HRG_M + HRG_M_s
#
# i=0
# for Gk in HRG_M:
#		 with open('dblp_hrg_'+str(i)+'.txt', 'w') as tmp:
#				 for e in Gk.edges():
#						 tmp.write(str(e[0]) + ' ' + str(e[1]) + '\n')
#		 i += 1
# """"
#
# print "PCFG"
#
# pHRG_M_s = PHRG.probabilistic_hrg(G, 5)
# pHRG_M = pHRG_M + pHRG_M_s
#
# i=0
# for Gk in pHRG_M:
#		 with open('dblp_phrg_'+str(i)+'.txt', 'w') as tmp:
#				 for e in Gk.edges():
#						 tmp.write(str(e[0]) + ' ' + str(e[1]) + '\n')
#		 i += 1
#
#
# for i in range(0, 3):
#		 # P = kronfit(G)
#		 print i,
#		 k = math.log(num_nodes, 2)
#		 Gk = product.kronecker2_random_graph(int(math.ceil(k)), P, directed=False)
#		 Gk.remove_edges_from(G.selfloop_edges())
#		 kron_M.append(Gk)
#
# i=0
# for Gk in kron_M:
#		 with open('dblp_kron_'+str(i)+'.txt', 'w') as tmp:
#				 for e in Gk.edges():
#						 tmp.write(str(e[0]) + ' ' + str(e[1]) + '\n')
#		 i += 1
#
# for i in range(0, 10):
#		 Gk = nx.expected_degree_graph(G.degree().values())
#		 Gk.remove_edges_from(Gk.selfloop_edges())
#
#		 with open('dblp_cl_'+str(i)+'.txt', 'w') as tmp:
#				 for e in Gk.edges():
#						 tmp.write(str(e[0]) + ' ' + str(e[1]) + '\n')
#
#
# HRG_M_s, degree = HRG.stochastic_hrg(G, 2)
# HRG_M = HRG_M + HRG_M_s
#
# i=0
# for Gk in HRG_M:
#		 with open('dblp_hrg_'+str(i)+'.txt', 'w') as tmp:
#				 for e in Gk.edges():
#						 tmp.write(str(e[0]) + ' ' + str(e[1]) + '\n')
#		 i += 1
# """""
#
# print "PCFG"
#
# pHRG_M_s = PHRG.probabilistic_hrg(G, 5)
# pHRG_M = pHRG_M + pHRG_M_s
#
# i=0
# for Gk in pHRG_M:
#		 with open('dblp_phrg_'+str(i)+'.txt', 'w') as tmp:
#				 for e in Gk.edges():
#						 tmp.write(str(e[0]) + ' ' + str(e[1]) + '\n')
#		 i += 1
#
#
# for i in range(0, 3):
#		 # P = kronfit(G)
#		 print i,
#		 k = math.log(num_nodes, 2)
#		 Gk = product.kronecker2_random_graph(int(math.ceil(k)), P, directed=False)
#		 Gk.remove_edges_from(G.selfloop_edges())
#		 kron_M.append(Gk)
#
# i=0
# for Gk in kron_M:
#		 with open('dblp_kron_'+str(i)+'.txt', 'w') as tmp:
#				 for e in Gk.edges():
#						 tmp.write(str(e[0]) + ' ' + str(e[1]) + '\n')
#		 i += 1
#
#
# df_g = metrics.external_rage(G)
# gcd_chunglu = []
# gcd_phrg = []
# gcd_hrg = []
# gcd_kron = []
# for chunglu_M_s in chunglu_M:
#		 df_chunglu = metrics.external_rage(chunglu_M_s)
#		 rgfd = metrics.tijana_eval_rgfd(df_g, df_chunglu)
#		 gcm_g = metrics.tijana_eval_compute_gcm(df_g)
#		 gcm_h = metrics.tijana_eval_compute_gcm(df_chunglu)
#		 gcd_chunglu.append(metrics.tijana_eval_compute_gcd(gcm_g, gcm_h))
# for HRG_M_s in HRG_M:
#		 df_hrg = metrics.external_rage(HRG_M_s)
#		 rgfd = metrics.tijana_eval_rgfd(df_g, df_hrg)
#		 gcm_g = metrics.tijana_eval_compute_gcm(df_g)
#		 gcm_h = metrics.tijana_eval_compute_gcm(df_hrg)
#		 gcd_hrg.append(metrics.tijana_eval_compute_gcd(gcm_g, gcm_h))
# for pHRG_M_s in pHRG_M:
#		 df_phrg = metrics.external_rage(pHRG_M_s)
#		 rgfd = metrics.tijana_eval_rgfd(df_g, df_phrg)
#		 gcm_g = metrics.tijana_eval_compute_gcm(df_g)
#		 gcm_h = metrics.tijana_eval_compute_gcm(df_phrg)
#		 gcd_phrg.append(metrics.tijana_eval_compute_gcd(gcm_g, gcm_h))
# for kron_M_s in kron_M:
#		 df_kron = metrics.external_rage(kron_M_s)
#		 rgfd = metrics.tijana_eval_rgfd(df_g, df_kron)
#		 gcm_g = metrics.tijana_eval_compute_gcm(df_g)
#		 gcm_h = metrics.tijana_eval_compute_gcm(df_kron)
#		 gcd_kron.append(metrics.tijana_eval_compute_gcd(gcm_g, gcm_h))
#
#
#
# print gcd_chunglu
# print gcd_hrg
# print gcd_phrg
# print gcd_kron
# print
# print
#
# exit()
#
# metrics.draw_degree_probability_distribution(G_M, chunglu_M, HRG_M, pHRG_M, kron_M)
# metrics.draw_network_value(G_M, chunglu_M, HRG_M, pHRG_M, kron_M)
# metrics.draw_hop_plot(G_M, chunglu_M, HRG_M, pHRG_M, kron_M)
# metrics.draw_kcore_decomposition(G_M, chunglu_M, HRG_M, pHRG_M, kron_M)
# metrics.draw_clustering_coefficients(G_M, chunglu_M, HRG_M, pHRG_M, kron_M)
# metrics.draw_assortativity_coefficients(G_M, chunglu_M, HRG_M, pHRG_M, kron_M)
# exit()
#
#
#
#
#
#
#
#
#
# er_G = nx.erdos_renyi_graph(num_nodes, .1)
# ws_G = nx.watts_strogatz_graph(num_nodes, 8, .1)
# nws_G = nx.newman_watts_strogatz_graph(num_nodes, 8, .1)
#
# graphs = [ba_G, er_G, ws_G, nws_G]
#
#
# for G in graphs:
#
#		 chunglu_M = []
#		 for i in range(0, samples):
#				 chunglu_M.append(nx.expected_degree_graph(G.degree()))
#
#
#
#
#		 gcd_chunglu = []
#		 gcd_phrg = []
#		 gcd_hrg = []
#		 gcd_kron = []
#
#
#
#		 #metrics.draw_diam_plot(G, [chunglu_M, HRG_M, pHRG_M, kron_M] )
#		 metrics.draw_degree_rank_plot(G, chunglu_M)
#		 metrics.draw_network_value(G, chunglu_M)
#
#
#
#
#
#
# Rnd = snap.TRnd()
# Graph = snap.GenRMat(1000, 2000, .6, .1, .15, Rnd)
# for EI in Graph.Edges():
#		 print "edge: (%d, %d)" % (EI.GetSrcNId(), EI.GetDstNId())
#
# print gcd, rgfd
def graph_name(fname):
	gnames= [x for x in os.path.basename(fname).split('.') if len(x) >3][0]
	if len(gnames):
		return gnames
	else:
		return gnames[0]

def read_load_graph(fname):
	df = Pandas_DataFrame_From_Edgelist([fname])[0]
	G	= nx.from_pandas_dataframe(df, source='src',target='trg')
	Gc = max(nx.connected_component_subgraphs(G), key=len)
	gname = graph_name(fname)
	Gc.name= gname
	return Gc

def compute_netstats_peek(origFullG, fbname, piikshl=False):
	for path, subdirs, files in os.walk("Results/"):
		shlfiles = [x for x in files if x.endswith("shl")]
	gname = [x for x in fbname.split('.') if len(x)>3][0]
	for f in shlfiles:
		#if not 'radoslaw' in f: continue
		if not gname in f: continue
		s = shelve.open("Results/"+f)
		if s.keys():
			for k,v in s.items():
				print '\t',k, len(v), "\n\t","-"*40
	print "[_]"


def compute_netstats(origFullG,fbname):
	''' 
	links: https://pymotw.com/2/shelve/
	'''
	for path, subdirs, files in os.walk("Results/"):
		shlfiles = [x for x in files if x.endswith("shl")]
	if 0: pp.pprint (shlfiles)
	gname = [x for x in fbname.split('.') if len(x)>3][0]
	for f in shlfiles:
		#if not 'radoslaw' in f: continue
		if not gname in f: continue
		s = shelve.open("Results/"+f)
		try:
			if s.keys():
				print 'netstats'
				pp.pprint((f, s.keys())) #existing = s['Akey1']
				for k,v in s.items():
					print k, len(v), "\n","-"*40
					metricx = ['gcd']
					metricx = ['degree','hops', 'clust', 'assort', 'kcore','eigen', 'gcd']
					metricx = ['degree','hops', 'clust', 'eigen', 'gcd']
					metrics.network_properties([origFullG], metricx, v, name=gname+"_"+str(k), out_tsv=False)

					"""
					print "degree ECDF -< optional >-"
					kcdf  = metrics.degree_ecdf(origFullG, v, gname+"_"+str(k)) 
					# print kcdf.apply(lambda x: '{}, {}, {}'.format(x.index, x[0], np.mean(x[1:])), axis=1)
					#print kcdf.head()
					kcdf['mu'] = kcdf.ix[:,1:].mean(axis=1)
					df = kcdf[['orig','mu']]
					df.to_csv("Results/kecdf_{}_{}.dat".format(gname,k), sep="\t", header=False, na_rep='nan')
					if 1: print df.to_string()
					"""
					#for r in kcdf.iterrows():
					#	print r[0], r[1]['orig'], np.mean(r[1][2:])
					#for p in metrics.SNDegree_CDF_ks_2samp(v, origFullG, gname+"_"+str(k)):
					#	x,y = [els.tolist() for els in p]# for x in els.tolist()]
				  #		print "{}, {}".format(x,y)

		finally:
			s.close()



def main(argsD):
	runs = argsD['runs']
	print 	
	print 'dataset: {}\nruns: {},'.format(argsD['orig'][0], runs), 
	G = read_load_graph(argsD['orig'][0])
	print "(V,E): {},{}".format(G.number_of_nodes(), G.number_of_edges())
	## if metrix
	if argsD['netstats']:
		compute_netstats(G, G.name)
		exit(0)

	if argsD['peek']:
		compute_netstats_peek(G, G.name, piikshl=True)
		exit(0)

	ofname = "Results/"+ G.name+ ".shl"
	# if argsD['rods']: ofname = ofname.split(".")[0] + "_rods.shl"
	database = shelve.open(ofname)
	

	if argsD['rods']:
		print '% --> Control Rods'
		start_time = time.time()
		HRG_M, degree = HRG.stochastic_hrg(G, runs)
		print("  %d, %s seconds ---" % (G.number_of_nodes(), time.time() - start_time))
		database['rods_hstars'] = HRG_M
	else:
		print '% --> PHRG'
		start_time = time.time()
		A = PHRG.probabilistic_hrg(G,runs) # returns a list of Hstar graphs
		# print("  --- Total %s seconds ---" % (time.time() - start_time))
		print("  %d, %s seconds ---" % (G.number_of_nodes(), time.time() - start_time))
		database['prob_hstars'] = A
	
	print 
	start_time = time.time()
	print '% --> CHLU'
	clgs = []
	z = G.degree().values()
	for i in range(runs):
		clgs.append(nx.expected_degree_graph(z))
	database['clgs'] = clgs
	print("  %d, %s seconds ---" % (G.number_of_nodes(),time.time() - start_time))
	# -- Kron Prod Graphs
	print '% --> Kron'
	start_time = time.time()
	database['kpgs'] = grow_graphs_using_krongen(G,gn=G.name,nbr_runs=runs)
	print("  %d, %s seconds ---" % (G.number_of_nodes(), time.time() - start_time))
	
	database.close()


	return

def get_parser ():
	parser = argparse.ArgumentParser(description='Pami compute the metrics HRG, PHRG, ' +\
			'CLGM, KPGM.\nexample: \n'+ \
			'  python pami.py --orig ./datasets/out.radoslaw_email_email --netstat')
	parser.add_argument('--rods', action='store_true', default=0, required=0,help="Edglst to Dimacs.")
	parser.add_argument('--netstats', action='store_true', default=0, required=0,
			help="Net Stats (metrics)")
	parser.add_argument('--peek', action='store_true', default=0, required=0,
			help="Peek at key/val pairs in the shl file")
	parser.add_argument('--orig', nargs=1, required=True, 
			help="Reference (edgelist) input file.")
	parser.add_argument('--runs', required=False, type=int, 
			help="Reference (edgelist) input file.")
	parser.add_argument('--version', action='version', version=__version__)
	return parser

if __name__ == '__main__':
	''' PAMI 
	arguments:
		rods - using the stochastic graph generation 
		otherwise, use the fixed size generation
		 '''
	parser = get_parser()
	args = vars(parser.parse_args())
	try:
		main(args)
	except Exception, e:
		print str(e)
		traceback.print_exc()
		sys.exit(1)
	sys.exit(0)
