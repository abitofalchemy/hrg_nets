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


def get_parser ():
	parser = argparse.ArgumentParser(description='Pami compute the metrics HRG, PHRG, ' +\
			'CLGM, KPGM.\nexample: \n'+ \
			'	python pami.py --orig ./datasets/out.radoslaw_email_email --netstat')
	parser.add_argument('--orig', nargs=1, required=True, 
			help="Reference (edgelist) input file.")
	parser.add_argument('--version', action='version', version=__version__)
	return parser

def hstar_nxto_tsv(G,gname, ix):
	print type(G), gname, ix
	import tempfile
	with tempfile.NamedTemporaryFile(dir='/tmp', delete=False) as tmpfile:
		tmp_fname = tmpfile	
		print tmp_fname
		nx.write_edgelist(G, tmp_fname, delimiter=",")
		
	return tmp_fname
	
def main(args):
	if not args['orig']: return 
	print args['orig'][0]

	with open(args['orig'][0], "rb") as f:
		c = cPickle.load(f)

	gname = graph_name(args['orig'][0])
	print type(c), len(c), gname
	## -- 
	proc_retval = []
	for j,gnx in enumerate(c):
		p = Process(target=hstar_nxto_tsv, args=(gnx,gname,j, )).start()
		proc_retval.append(p)
		break

	print "^^^"
	print proc_retval

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
