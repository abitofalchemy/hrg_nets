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
import os#, sys
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import networkx as nx
from pami import graph_name
from multiprocessing import Process
from glob import glob

def get_parser ():
	parser = argparse.ArgumentParser(description='Pami compute the metrics HRG, PHRG, ' +\
			'CLGM, KPGM.\nexample: \n'+ \
			'	python pami.py --orig ./datasets/out.radoslaw_email_email --netstat')
	parser.add_argument('--orig', nargs=1, required=True, 
			help="Reference (edgelist) input file.")
	parser.add_argument('--stats', action='store_true', default=0, required=0, help="print xphrg mcs tw")

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
	files = glob("Results_Graphlets/{}*.macro".format(gname))
	print files

	return 


def main(args):
	if not args['orig']: return 
	print args['orig'][0]
	if args['stats']:
		graphlet_stats(args)
		exit()

	with open(args['orig'][0], "rb") as f:
		c = cPickle.load(f)

	gname = graph_name(args['orig'][0])
	print type(c), len(c), gname
	if isinstance(c, dict):
		if len(c.keys()) == 1:
			c = c.values()[0]
		else:
			print c.keys()

	## -- 
	for j,gnx in enumerate(c):
		if isinstance (gnx, list):
			gnx = gnx[0]
		p = Process(target=hstar_nxto_tsv, args=(gnx,gname,j, )).start()
		print p


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
