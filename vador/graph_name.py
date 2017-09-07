import os 

def graph_name(fname):
  gnames= [x for x in os.path.basename(fname).split('.') if len(x) >3][0]
  if len(gnames):
    return gnames
  else:
    return gnames[0]

