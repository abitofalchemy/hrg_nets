#!/usr/bin/env python
import networkx as nx
import os, sys
import load_edgelist_from_dataframe as dfe
from numpy import mean 


if len(sys.argv)>1:
  ifile = sys.argv[1]
else:
  exit(1)

dfs = dfe.Pandas_DataFrame_From_Edgelist([ifile])
df = dfs[0]

try:
    g = nx.from_pandas_dataframe(df, 'src', 'trg',edge_attr=['ts'])
except  Exception, e:
    g = nx.from_pandas_dataframe(df, 'src', 'trg')

if df.empty:
  g = nx.read_edgelist(ifile,comments="%")
print g.number_of_nodes(), g.number_of_edges(), "%.2f" % mean(g.degree().values()), '\t',
print os.path.basename(ifile)

