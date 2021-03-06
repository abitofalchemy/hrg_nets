#!/usr/bin/env python
__version__="0.1.0"
__author__ = 'tweninge'
__author__ = 'saguinag'


# Version: 0.1.0 Forked from cikm_experiments.py
#
# ToDo:
#

import cPickle
import subprocess
import traceback
import argparse
import os
from collections import Counter
from random import sample

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import networkx as nx
import pandas as pd
import pprint as pp
from pami import graph_name
from multiprocessing import Process
from glob import glob

def get_parser ():
	parser = argparse.ArgumentParser(description='Pami compute the metrics HRG, PHRG, ' +\
			'CLGM, KPGM.\nexample: \n'+ \
			'	python pami.py --orig ./datasets/out.radoslaw_email_email --netstat')
	parser.add_argument('--orig', nargs=1, required=True, help="Reference (edgelist) input file.")
	parser.add_argument('--stats', action='store_true', default=0, required=0, \
						help="print xphrg mcs tw")
	parser.add_argument('--version', action='version', version=__version__)
	return parser


def run_pgd_on_edgelist(fname,graph_name):
	import platform
	if platform.system() == "Linux":
		args = ("bin/linux/pgd", "-f",  "{}".format(fname), "--macro", "Results_Graphlets/{}_{}.macro".format(graph_name, \
			os.path.basename(fname)))
		print args
	else:
		args = ("bin/macos/pgd", "-f {}".format(fname), "--macro Results_Graphlets/{}_{}.macro".format(graph_name, fname))

	pgdout = ""
	while not pgdout:
		popen = subprocess.Popen(args, stdout=subprocess.PIPE)
		popen.wait()
		output = popen.stdout.read()
		pgdout = output.split('\n')
	print "[o]"
	return 

def hops(all_succs, start, level=0, debug=False):
    if debug: print("level:", level)

    succs = all_succs[start] if start in all_succs else []
    if debug: print("succs:", succs)

    lensuccs = len(succs)
    if debug: print("lensuccs:", lensuccs)
    if debug: print()
    if not succs:
        yield level, 0
    else:
        yield level, lensuccs

    for succ in succs:
        # print("succ:", succ)
        for h in hops(all_succs, succ, level + 1):
            yield h

def get_graph_hops(graph, num_samples):
    c = Counter()
    for i in range(0, num_samples):
        node = sample(graph.nodes(), 1)[0]
        b = nx.bfs_successors(graph, node)

        for l, h in hops(b, node):
            c[l] += h

    hopper = Counter()
    for l in c:
        hopper[l] = float(c[l]) / float(num_samples)
    print hopper
    return hopper

def hstar_nxto_tsv(G,gname, ix):
	import tempfile
	with tempfile.NamedTemporaryFile(dir='/tmp', delete=False) as tmpfile:
		tmp_fname = tmpfile	
		nx.write_edgelist(G, tmp_fname, delimiter=",")
		if os.path.exists(tmp_fname.name):			
			run_pgd_on_edgelist(tmp_fname.name, gname)
		else:
			print "{} file does not exists".format(tmp_fname)

	return tmp_fname
	

def graphlet_stats(args):
	gname = graph_name(args['orig'][0])
	print "==>", gname
	rows_2keep =["total_3_tris",
				 "total_2_star",
				 "total_4_clique",
				 "total_4_chordcycle",
				 "total_4_tailed_tris",
				 "total_4_cycle",
				 "total_3_star",
				 "total_4_path"]
	files = glob("Results_Graphlets/{}*.macro".format(gname))
	if len(files) ==0:
		print "!!!> no macro files"
		return 
	mdf = pd.DataFrame()
	for j,f in enumerate(files):
		df = pd.read_csv(f, delimiter=r" ", header=None)
		df.columns=['a','b','c']
		df['c'] = df['c'].astype('float32')
		df = df.drop('b', axis=1)
		local_df = []
		for r in rows_2keep:
			local_df.append(df[df['a'] == r].values[0])
		local_df = pd.DataFrame(local_df)
		local_df.columns = ['a','c']

		if j == 0:	mdf = local_df
		mdf = mdf.merge(local_df, on='a')
	df = pd.DataFrame(list(mdf['a'].values))
	mdf = mdf.drop('a', 1)
	
	df['mu'] = mdf.apply(lambda x: x[1:].mean(), axis=1)
	df['st'] = mdf.apply(lambda x: x[1:].std(), axis=1)
	df['se'] = mdf.apply(lambda x: x[1:].sem(), axis=1)
	df['dsc'] = mdf.apply(lambda x: x[1:].count(), axis=1)
	print df

	return


def main(args):
	if not args['orig']: return 
	print "==>", args['orig'][0]
	if args['stats']:
		graphlet_stats(args)
		exit()
	
	with open(args['orig'][0], "rb") as f:
		c = cPickle.load(f)
	print "==> [1]"
	gname = graph_name(args['orig'][0])
	print "==>", type(c), len(c), gname
	if isinstance(c, dict):
		if len(c.keys()) == 1:
			c = c.values()[0]
		else:
			print c.keys()

	## -- 
	for j,gnx in enumerate(c):
		if isinstance (gnx, list):
			gnx = gnx[0]
		Process(target=get_graph_hops, args=(gnx,100, )).start()


	return 


if __name__ == '__main__':
	''' PAMI 
	arguments:
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
