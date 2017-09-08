import os
import dill
import pprint as pp
import multiprocessing as mp
import load_edgelist	as nxg

results = {}
"""
	Links:
	[1] https://spectraldifferences.wordpress.com/2014/03/02/recursively-finding-files-in-python-using-the-find-command/
	[2] https://stackoverflow.com/questions/25708026/loading-a-pkl-file-using-dill
	def get_parser ():
	parser = argparse.ArgumentParser(description='Pami compute the metrics HRG, PHRG, '
	+ 'CLGM, KPGM.\nexample: \n'
	+ '       python pami.py --orig ./datasets/out.radoslaw_email_email --netstat')
	parser.add_argument('--orig', nargs=1, required=True, help="Reference")
	echo_welcome()
	"""


def edglist_name_graph(fname):
	print fname
	return {nxg.graph_name(fname): gnx.load_edgelist(fname)}
def collect_results(result):
	#results.extend(result)
	# https://stackoverflow.com/questions/8930915/append-dictionary-to-a-
	results.update(result)

def load_pami_datasets(edge_lst_fnames):
	p = mp.Pool(processes=mp.cpu_count())
	for f in edge_lst_fnames:
		p.apply_async(nxg.load_edgelist, args= (f,), callback=collect_results)
	p.close()
	p.join()
	if 0: pp.pprint(results) 
	return results

def load_graphs_nxobjects(mango_file):
	print ("-- mango exists, loading pickle into `graphs_d` ...")
	with open(mango_file, 'rb') as in_strm:
		graphs_d = dill.load(in_strm)
	return graphs_d

def gen_graphs_obj_dict(mango_file,local_edge_lst_f):
	dat_sets_fnames = open(local_edge_lst_f).readlines()
	dat_sets_fnames = [f.rstrip("\r\n") for f in dat_sets_fnames]
	graphs_d = load_pami_datasets(dat_sets_fnames)
	with open(mango_file, 'wb') as out_strm:
		dill.dump(graphs_d, out_strm)
	return 

def echo_welcome():
	print "="*80
	print "Point us toward towards where you edgelist (out.*) are located"
	print "-"*80

def gen_datasets_edgelist_file():
	datasets_path  = raw_input("Enter your datasets path: ")   # Python 2.x
	# args = ("find", datasets_path, "-iname 'out.*'", "-type f")
	import os, fnmatch
 
	pattern = 'out.socio*'
	fileList = []

	# Walk through directory
	# ref[1]
	for dName, sdName, fList in os.walk(datasets_path):
		for fileName in fList:
			if fnmatch.fnmatch(fileName, pattern): # Match search string
				fileList.append(os.path.join(dName, fileName))
	# save to file
	with open(EDGELISTS, 'w') as thefile:
		for item in fileList:
			print>>thefile, item
	if os.path.exists(EDGELISTS):
		print "-- Wrote datasets edgelist file"


EDGELISTS = "./datasets/tmp_local_datasets_edgelist.txt"
GRPHSDICT = "./datasets/mango.pickle"

def graphs_dict():
	if not os.path.exists(EDGELISTS):
		echo_welcome()
		gen_datasets_edgelist_file()


	if not os.path.exists(GRPHSDICT):
		# ref[2]
		grphs_d = gen_graphs_obj_dict(GRPHSDICT,EDGELISTS)
		print ("-- Graphs as nx.obj in grphs_d")
	else:
		print "-- loading pickle of local datasets"
		grphs_d = load_graphs_nxobjects(GRPHSDICT)

	print ("-----------------")
	if 1: pp.pprint (grphs_d)
	return grphs_d
