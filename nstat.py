import os 
import networkx as nx
import shelve
from load_edgelist_from_dataframe import Pandas_DataFrame_From_Edgelist


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

argsD = {'orig': ['../datasets/out.amazon0312']}
fbname = os.path.basename(argsD['orig'][0])
G = read_load_graph(argsD['orig'][0])
print "(V,E): {},{}".format(G.number_of_nodes(), G.number_of_edges())
## if metrix

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
