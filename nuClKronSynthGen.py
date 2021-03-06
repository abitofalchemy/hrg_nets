#!/usr/bin/env python

__version__ = "0.1.0"
__author__ = 'saguinag'

import subprocess#, shlex
import math
import networkx as nx
import numpy as np
import sys
import os
import re
import cPickle
import time
import load_edgelist_from_dataframe as tdf
import argparse, traceback
from multiprocessing import Process
import os
import multiprocessing as mp

debug = DBG = False
results = []	 
chunglu_GM = []


#~#~#~#~#~##~#~#~#~#~##~#~#~#~#~##~#~#~#~#~##~#~#~#~#~##~#~#~#~#~##~#~#~#~#~##~#~#~#~#~##~#~#~#~#~##~#~#~#~60
def get_parser ():
	parser = argparse.ArgumentParser(description='Infer a model given a graph (derive a model)')
	parser.add_argument('--orig', required=True, nargs=1, help='Filename of edgelist graph')
	parser.add_argument('--synths', help='Generate CL + Kron graphs', action='store_true')
	parser.add_argument('--version', action='version', version=__version__)
	return parser

def collect_results(result):
		results.extend(result)

def collect_chunglu_GM(result):
		chunglu_GM.extend(result)

def grow_graphs_using_krongen(graph, gn, P, k, recurrence_nbr=1, graph_vis_bool=False, nbr_runs = 1):
	"""
	grow graph using krongen given orig graph, gname, and # of recurrences
	Returns
	-------
	nth graph --<kpgm>--
	"""
	import math
	from pami import kronfit
	from os import environ
	import subprocess

	tsvGraphName = "./tmp_{}kpgraph.tsv".format(gn)
	#	tmpGraphName = "/tmp/{}kpgraph.tmp".format(gn)
	
	#	if environ['HOME'] == '/home/saguinag':
	#		args = ("time/bin/linux/krongen", "-i:{}".format(tsvGraphName),"-n0:2", "-m:\"0.9 0.6; 0.6 0.1\"", "-gi:5")
	#	elif environ['HOME'] == '/Users/saguinag':
	#		args = ("time/bin/mac/krongen", "-i:{}".format(tsvGraphName),"-n0:2", "-m:\"0.9 0.6; 0.6 0.1\"", "-gi:5")
	#	else:
	#		args = ('./kronfit.exe -i:tmp.txt -n0:2 -m:"0.9 0.6; 0.6 0.1" -gi:5')

	kp_graphs = []
	

	M = '-m:"{} {}; {} {}"'.format(P[0][0], P[0][1], P[1][0], P[1][1])
	if environ['HOME'] == '/home/saguinag':
		args = ("bin/linux/krongen", "-o:"+tsvGraphName, M, "-i:{}".format(k))
	elif environ['HOME'] == '/Users/saguinag':
		print tsvGraphName
		args = ("bin/macos/krongen", "-o:"+tsvGraphName, M, "-i:{}".format(k))
	else:
		args = ('./krongen.exe -o:{} '.format(tsvGraphName) +M +'-i:{}'.format(k+1))
	for i in range(nbr_runs):
		popen = subprocess.Popen(args, stdout=subprocess.PIPE)
		popen.wait()
		#output = popen.stdout.read()

		if os.path.exists(tsvGraphName):
			KPG = nx.read_edgelist(tsvGraphName, nodetype=int)
		else:
			print "!! Error, file is missing"

		for u,v in KPG.selfloop_edges():
			KPG.remove_edge(u, v)
		kp_graphs.append( KPG )
		if DBG: 
			print 'Avg Deg:', nx.average_degree_connectivity(graph)
			import phoenix.visadjmatrix as vis
			# vis.draw_sns_adjacency_matrix(graph)
			vis.draw_sns_graph(graph)

	return kp_graphs # returns a list of kp graphs


def kronfit(G):
	"""

	Notes:
	23May16: changes to handle kronfit
	"""

	from os import environ

	with open('tmp.txt', 'w') as tmp:
			for e in G.edges():
					tmp.write(str(e[0]) + ' ' + str(e[1]) + '\n')

	if environ['HOME'] == '/home/saguinag':
		args = ("bin/linux/kronfit", "-i:tmp.txt","-n0:2", "-m:\"0.9 0.6; 0.6 0.1\"", "-gi:5")
	elif environ['HOME'] == '/Users/saguinag':
		args = ("bin/macos/kronfit", "-i:tmp.txt","-n0:2", "-m:\"0.9 0.6; 0.6 0.1\"", "-gi:5")
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
	timer = Timer(6, kill_proc, [proc])
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


def get_kron_synthgraphs(origG, kron_GM,j):
	k = int(math.log(origG.number_of_nodes(),2))+1 # Nbr of Iterations
	if 0: print 'k:',k,'n',origG.number_of_nodes()
	print "	--- get_kron_synthgraphs",j
	P = kronfit(origG) #[[0.9999,0.661],[0.661,		 0.01491]]
	kron_GM.append(grow_graphs_using_krongen(origG, origG.name, P, k))
	return kron_GM

def main(args):
	# load orig file into DF and get the dataset name into g_name
	datframes = tdf.Pandas_DataFrame_From_Edgelist(args['orig'])
	df = datframes[0]
	g_name = [x for x in os.path.basename(args['orig'][0]).split('.') if len(x) > 3][0]
	t_start = time.time()
	## --
	if df.shape[1] == 4:
		G = nx.from_pandas_dataframe(df, 'src', 'trg', edge_attr=True)	# whole graph
	elif df.shape[1] == 3:
		G = nx.from_pandas_dataframe(df, 'src', 'trg', ['ts'])	# whole graph
	else:
		G = nx.from_pandas_dataframe(df, 'src', 'trg')
	G.name = g_name
	print "==> read in graph took: {} seconds".format(time.time() - t_start)
		
	G.remove_edges_from(G.selfloop_edges())
	giant_nodes = max(nx.connected_component_subgraphs(G), key=len)
	G = nx.subgraph(G, giant_nodes)
	print "Done cleaning Graph"

	if os.path.exists("Results/{}_clgms.pickle".format(g_name)):
		print "--[^]--"
	else:
		chunglu_GM = []
		print (" --< chung lu graph >--")
		t_start = time.time()
		z = G.degree().values()
		for i in range(10):
			chunglu_GM.append(nx.expected_degree_graph(z))

		print "  ",(chunglu_GM[:4])
		print "  ==> chunglu_GM took: {} seconds".format(time.time() - t_start)
		print len(chunglu_GM)
		with open(r"Results/{}_clgms.pickle".format(g_name), "wb") as output_file:
			cPickle.dump(chunglu_GM, output_file)
		if os.path.exists(r"Results/{}_clgms.pickle".format(g_name)): 
			print "File saved", r"Results/{}_clgms.pickle".format(g_name)

	# --<
	# --< Kronecker product graph >--
	# --<
	if os.path.exists("Results/{}_kpgms.pickle".format(g_name)):
		print "--[^]--"
		exit()
	else:
		nu_main(G)
	
	return 
	"""
	t_start = time.time()
	[get_kron_synthgraphs (G,kronprd_GM, i) for i in range(0,6)]
	#	for i in range(6):
	#		Process(target=get_kron_synthgraphs, args=(G,kronprd_GM)).start()
	#		all_proc.append(p)
	#	for p in kronprd_GM:
	#		p.join()
	#	print type (p)
	#	print len(p)
	print "Using: get_kron_synthgraphs:	%s seconds" % (time.time()-t_start)
	print len(kronprd_GM), type(kronprd_GM)
	with open(r"Results/{}_kpgms.pickle".format(g_name), "wb") as output_file:
		cPickle.dump(kronprd_GM, output_file)
	if os.path.exists(r"Results/{}_kpgms.pickle".format(g_name)): print "File saved"
	"""


def testFunc(file):
		result = []
		print "Working in Process #%d" % (os.getpid())
		# This is just an illustration of some logic. This is not what I'm
		# actually doing.
		with open(file, 'r') as f:
				for line in f:
						if 'dog' in line:
								result.append(line)
		return result


def nu_main(nxg):
	t_start = time.time()
	p = mp.Pool(processes=20)
	kronprd_GM =[]
	for j in range(10):
		#get_kron_synthgraphs (G,kronprd_GM, i) for i in range(0,6)]
		p.apply_async(get_kron_synthgraphs, args=(nxg,kronprd_GM,j ), callback=collect_results)
	p.close()
	p.join()
	if 0: print(results)
	print "Using: get_kron_synthgraphs: %s seconds" % (time.time()-t_start)

	with open(r"Results/{}_kpgms.pickle".format(nxg.name), "wb") as output_file:
		cPickle.dump(results, output_file)
	if os.path.exists(r"Results/{}_kpgms.pickle".format(nxg.name)): print "File saved"


if __name__ == '__main__':
	parser = get_parser()
	args = vars(parser.parse_args())
	
	try:
		main(args)
	except	Exception, e:
		print 'ERROR, UNEXPECTED SAVE PLOT EXCEPTION'
		print str(e)
		traceback.print_exc()
		os._exit(1)
	sys.exit(0)
